# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
import random
from collections import namedtuple
from datetime import datetime
import copy

# PennyLane
import pennylane as qml

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm.autonotebook import tqdm


# OpenAI Gym import
import gymnasium as gym
from gymnasium import spaces

# from distributed import init_distributed

# Fix seed for reproducibility
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed);

# To get smooth animations on Jupyter Notebooks.
# Note: these plotting function are taken from https://github.com/ageron/handson-ml2
import matplotlib as mpl

import tianshou as ts
from tianshou.policy import DQNPolicy, QRDQNPolicy, RainbowPolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.utils.net.common import DataParallelNet

import argparse

# def get_arguments():
#     """
#     handle arguments from commandline.
#     some other hyper parameters can only be changed manually (such as model architecture,dropout,etc)
#     notice some arguments are global and take effect for the entire three phase training process, while others are determined per phase
#     """
#     parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
#     # DDP configs:
#     parser.add_argument('--world_size', default=-1, type=int, 
#                         help='number of nodes for distributed training')
#     parser.add_argument('--rank', default=-1, type=int, 
#                         help='node rank for distributed training')
#     parser.add_argument('--local_rank', default=-1, type=int, 
#                         help='local rank for distributed training')
#     parser.add_argument('--dist_backend', default='nccl', type=str, 
#                         help='distributed backend')
#     parser.add_argument('--init_method', default='env', type=str, choices=['file','env'], help='DDP init method')
#     parser.add_argument('--distributed', default=False)
    
#     args = parser.parse_args()
#     return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="IGT")
    parser.add_argument("--log_num", type=str, default='0')    
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-quantiles", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.)
    parser.add_argument("--v-max", type=float, default=10.)    
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--qnn-layers", type=int, default=3)
    parser.add_argument("--data-reupload", action="store_false", default=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()

args=get_args()


def encode(n_qubits, inputs):
    for wire in range(n_qubits):
        qml.RX(inputs[wire], wires=wire)


def layer(n_qubits, y_weight, z_weight):
    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)
    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])


def measure(n_qubits):
    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]


def get_model(n_qubits, n_layers, data_reupload):
    dev = qml.device("default.qubit", wires=n_qubits)
    shapes = {
        "y_weights": (n_layers, n_qubits),
        "z_weights": (n_layers, n_qubits)
    }

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            if (layer_idx == 0) or data_reupload:
                encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return measure(n_qubits)

    model = qml.qnn.TorchLayer(circuit, shapes)

    return model


class QuantumQRDQN(nn.Module):
    def __init__(self, n_qubits, n_actions, n_layers, w_input, w_output, data_reupload, num_quantiles):
        super(QuantumQRDQN, self).__init__()
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.w_input = w_input
        self.w_output = w_output
        self.data_reupload = data_reupload
        self.num_quantiles=num_quantiles
        self.q_layers = get_model(n_qubits=self.n_qubits,
                                  n_layers=n_layers,
                                  data_reupload=data_reupload)
        if w_input:
            self.w_input2 = Parameter(torch.Tensor(self.n_qubits))
            nn.init.normal_(self.w_input2, mean=0.)
        else:
            self.register_parameter('w_input', None)
        if w_output:
            self.w_output2 = Parameter(torch.Tensor(self.n_actions))
            nn.init.normal_(self.w_output2, mean=90.)
        else:
            self.register_parameter('w_output', None)
        
        # Quantile Regression Layer
        self.fc_quantiles = nn.Sequential(
            nn.Linear(np.prod(self.n_actions), args.neurons), nn.BatchNorm1d(args.neurons), nn.ReLU(),
            nn.Linear(args.neurons, args.neurons), nn.BatchNorm1d(args.neurons), nn.ReLU(),
            nn.Linear(args.neurons, args.neurons), nn.BatchNorm1d(args.neurons), nn.ReLU(),
            nn.Linear(args.neurons, np.prod(self.n_actions) * num_quantiles)
            )

    def forward(self, inputs, **kwargs):
      batch_size = inputs.shape[0]  # Get the batch size
      outputs = []

      for i in range(batch_size):
        input_i = inputs[i]  # Get the i-th input in the batch
        input_i = torch.tensor(input_i, dtype=torch.float32)  # Convert input_i to a PyTorch tensor
        if self.w_input2 is not None:
            input_i = input_i * self.w_input2
        input_i = torch.atan(input_i)
        output_i = self.q_layers(input_i)
        output_i = (1 + output_i) / 2
        outputs.append(output_i)

      outputs = torch.stack(outputs)  # Stack outputs along the batch dimension

      if self.w_output2 is not None:
        outputs = outputs * self.w_output2
      else:
        outputs = 90 * outputs
        outputs = outputs.view(-1, self.n_qubits * 2)
 
      logits = self.fc_quantiles(outputs)
      return logits.view(batch_size, -1, self.num_quantiles), None

    
    def __deepcopy__(self, memodict={}):
        # Target Network: Create a new instance of the class
        new_instance = QuantumQRDQN(n_qubits = self.n_qubits,
                                      n_actions = self.n_actions,
                                      n_layers = self.n_layers,
                                      w_input = self.w_input,
                                      w_output = self.w_output,
                                      data_reupload = self.data_reupload,
                                      num_quantiles=self.num_quantiles)
        
        # Copy the fully connected layers for quantile regression
        new_instance.fc_quantiles = copy.deepcopy(self.fc_quantiles, memodict)
        
        # Assign the quantum parts after copying
        new_instance.q_layers = copy.deepcopy(self.q_layers, memodict)

        return new_instance

filename= '/scratch/connectome/justin/wi_100.csv'
reward_data = pd.read_csv(filename)
filename2= '/scratch/connectome/justin/lo_100.csv'
loss_data = pd.read_csv(filename2)
filename3= '/scratch/connectome/justin/choice_100.csv'
choice_data = pd.read_csv(filename3)

new_columns = ['subject']+[f'trial_{i}' for i in range(1, 101)]
reward_data.columns = new_columns
loss_data.columns = new_columns
choice_data.columns = new_columns
net_reward_data = reward_data.copy()
net_reward_data.iloc[:, 1:101] = loss_data.iloc[:, 1:101].add(reward_data.iloc[:, 1:101])

choices_df = choice_data.copy()
choices_df.iloc[:, 1:101] = choice_data.iloc[:, 1:101]-1

# Number of choices (A, B, C, D)
num_choices = 4

# Extract subjects and trial columns
subjects = choices_df['subject'].values
trial_columns = [col for col in choices_df.columns if col != 'subject']

# Create an empty 4D numpy array to store the state representations
state_array = np.zeros((len(subjects), len(trial_columns), num_choices))

# Iterate through subjects and trials to calculate the state representation
for i, subject in enumerate(subjects):
    for j, trial_column in enumerate(trial_columns):
        choice = choices_df.loc[choices_df['subject'] == subject, trial_column].values[0]
        reward = net_reward_data.loc[net_reward_data['subject'] == subject, trial_column].values[0]
        state_array[i, j, choice] = reward

# Calculate average rewards for each choice across trials for each subject
for i in range(len(state_array)):
  for j in range(len(state_array[i])):
    acc_rewards = state_array
    if j>0:
      acc_rewards[i][j] = state_array[i][j-1]+state_array[i][j]


class IGTEnv(gym.Env):
    def __init__(self, choices_df, acc_rewards, seed):
      # choices_df: Iowa Gambling Task Choices data: pandas dataframe
      # acc_rewards: Iowa Gambling Task Accumulated Net Rewards data: numpy array
        super(IGTEnv, self).__init__()
        self.choices_data = choices_df
        self.rewards_data = acc_rewards
        self.subjects = np.unique(choices_df['subject'])
        self.trials = [f'trial_{i}' for i in range(1, len(choices_df.columns))]

        # Observation space: 4-dimensional vector (accumulated rewards for A, B, C, D)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # Action space: discrete actions for choosing decks A, B, C, D
        self.action_space = spaces.Discrete(4)

        self.current_subject_idx = 0
        self.current_trial_idx = 0

    def reset(self):
        # Reset the environment to start a new episode
        self.current_subject_idx = 0
        self.current_trial_idx = 0
        return self._get_observation(), {}

    def step(self, action):
        # Determine reward based on choice.
        if action in [0, 1]:  # if A or B is chosen
            reward = -1
        elif action in [2, 3]:  # if C or D is chosen
            reward = 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Move to the next trial
        self.current_trial_idx += 1
        if self.current_trial_idx >= len(self.trials)-1:
            # Move to the next subject if all trials are completed
            self.current_subject_idx += 1
            self.current_trial_idx = 0

        # Check if all subjects and trials are completed
        done = self.current_subject_idx >= len(self.subjects)-1
        truncated = False  # or implement your own logic to handle truncation

        # Return the next state, reward, done, and additional info
        return self._get_observation(), reward, done, truncated, {}

    def _get_observation(self):
        state = self.rewards_data[self.current_subject_idx][self.current_trial_idx]
        return state

    def seed(self, seed):
        np.random.seed(seed)
    
env = IGTEnv(choices_df, acc_rewards, seed=2023)

train_envs = ts.env.DummyVectorEnv([lambda: IGTEnv(choices_df, acc_rewards, seed=2023) for _ in range(args.training_num)])
test_envs = ts.env.DummyVectorEnv([lambda: IGTEnv(choices_df, acc_rewards, seed=2023) for _ in range(args.test_num)])

# Use your defined network
state_shape = env.observation_space.shape[0]  # equivalent to 4 for IGT
action_shape = env.action_space.n  # equivalent to 4 for IGT

net = QuantumQRDQN(n_qubits=state_shape, n_actions=action_shape, n_layers=args.qnn_layers, w_input=True, w_output=True, 
                     data_reupload=args.data_reupload, num_quantiles=args.num_quantiles)
# net = net.to(args.device)
optim = torch.optim.RMSprop(net.parameters(), lr=args.lr)
policy = QRDQNPolicy(net, optim, discount_factor=args.gamma, 
                     num_quantiles=args.num_quantiles,
                     estimation_step=args.n_step,
                     target_update_freq=args.target_update_freq, is_double=True)
# policy = policy.to(args.device)

buffer = VectorReplayBuffer(total_size=args.buffer_size, buffer_num=args.training_num)  # max size of the replay buffer
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)


from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
writer = SummaryWriter(f'log_{args.log_num}/PennyLane_{args.task}_Quantum_QRDQN')
logger = TensorboardLogger(writer)

# Start training
result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=args.epoch,  # maximum number of epochs
    step_per_epoch=args.step_per_epoch,  # number of steps per epoch
    step_per_collect=args.step_per_collect,  # number of steps per data collection
    update_per_step=args.update_per_step,
    episode_per_test=100,  # number of episodes per test
    batch_size=args.batch_size,  # batch size for updating model
    train_fn=lambda epoch, env_step: policy.set_eps(args.eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(args.eps_test),
    logger=logger)

print(f'Finished training! Use {result["duration"]}')

path = f'/scratch/connectome/justin/log_{args.log_num}/PennyLane_{args.task}_Quantum_QRDQN_{args.log_num}.pth'
torch.save(policy.state_dict(), path)

policy.load_state_dict(torch.load(path))
policy.eval()
result = test_collector.collect(n_episode=100, render=False)
print("Quantum QRDQN Final reward: {}, length: {}".format(result["rews"].mean(), result["lens"].mean()))


path = f'/scratch/connectome/justin/log_{args.log_num}/PennyLane_{args.task}_Quantum_QRDQN_{args.log_num}.hdf5'
ts.data.ReplayBuffer.save_hdf5(buffer, path)

load_buffer= ts.data.ReplayBuffer.load_hdf5(path)

buffer_obs = buffer.get(index=10, key='obs', stack_num=args.buffer_size)
buffer_obs_next = buffer.get(index=10, key='obs_next', stack_num=args.buffer_size)
buffer_rew = buffer.get(index=10, key='rew', stack_num=args.buffer_size)
buffer_act = buffer.get(index=10, key='act', stack_num=args.buffer_size)

buffer_df = pd.DataFrame(buffer_obs, columns=['DeckA', 'DeckB', 'DeckC', 'DeckD'])
buffer_df2 = pd.DataFrame(buffer_obs_next, columns=['DeckA2', 'DeckB2', 'DeckC2', 'DeckD2'])
buffer_df3 = pd.DataFrame(buffer_rew, columns=['Reward'])
buffer_df4 = pd.DataFrame(buffer_act, columns=['Action'])

buffer_res = pd.concat([buffer_df, buffer_df2], axis=1)
buffer_res = pd.concat([buffer_res, buffer_df3], axis=1)
buffer_res = pd.concat([buffer_res, buffer_df4], axis=1)

buffer_res.to_csv(f'/scratch/connectome/justin/log_{args.log_num}/PennyLane_{args.task}_Quantum_QRDQN_{args.log_num}.csv')

