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

# from distributed import init_distributed

# Fix seed for reproducibility
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed);

# To get smooth animations on Jupyter Notebooks.
# Note: these plotting function are taken from https://github.com/ageron/handson-ml2
import matplotlib as mpl

import tianshou as ts
from tianshou.policy import RainbowPolicy
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
    parser.add_argument("--task", type=str, default='CartPole-v1')
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
    return [
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))
    ]


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

    @qml.qnode(dev, interface='torch')
    def circuit_entropy(inputs, y_weights, z_weights):
        for layer_idx in range(n_layers):
            if (layer_idx == 0) or data_reupload:
                encode(n_qubits, inputs)
            layer(n_qubits, y_weights[layer_idx], z_weights[layer_idx])
        return qml.vn_entropy(wires=[0])
    
    model = qml.qnn.TorchLayer(circuit, shapes)    
    entropy = qml.qnn.TorchLayer(circuit_entropy, shapes)
    return model, entropy

entropy_out = []


class QuantumRainbow(nn.Module):
    def __init__(self, n_qubits, n_actions, n_layers, w_input, w_output, data_reupload, num_quantiles):
        super(QuantumRainbow, self).__init__()
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.w_input = w_input
        self.w_output = w_output
        self.data_reupload = data_reupload
        self.num_quantiles = num_quantiles
        self.q_layers, self.entropy = get_model(n_qubits=self.n_qubits,
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

        # Additional fully connected layers for Distributional RL (QRDQN) and Dueling architecture (advantage)
        self.fc_quantiles = nn.Sequential(
            NoisyLinear(self.n_actions, args.neurons), nn.ReLU(),
            NoisyLinear(args.neurons, args.neurons), nn.ReLU(),
            NoisyLinear(args.neurons, num_quantiles)
        )
        self.fc_advantage = nn.Sequential(
            NoisyLinear(self.n_actions, args.neurons), nn.ReLU(),
            NoisyLinear(args.neurons, args.neurons), nn.ReLU(),
            NoisyLinear(args.neurons, self.n_actions*num_quantiles)
        )

    def forward(self, inputs, **kwargs):
      batch_size = inputs.shape[0]  # Get the batch size
      outputs = []
      entropy = []
    
      for i in range(batch_size):
        input_i = inputs[i]  # Get the i-th input in the batch
        input_i = torch.tensor(input_i, dtype=torch.float32)  # Convert input_i to a PyTorch tensor
        if self.w_input2 is not None:
            input_i = input_i * self.w_input2
        input_i = torch.atan(input_i)
        entropy_i = self.entropy(input_i)
        output_i = self.q_layers(input_i)
        output_i = (1 + output_i) / 2
        outputs.append(output_i)
        entropy.append(entropy_i)

      outputs = torch.stack(outputs)  # Stack outputs along the batch dimension
      # entropy_out = torch.stack(entropy)
      entropy_out.append(entropy)

      if self.w_output2 is not None:
        outputs = outputs * self.w_output2
      else:
        outputs = 90 * outputs
        outputs = outputs.view(-1, self.n_qubits * 2)

      # Distributional RL: Compute quantiles
      quantiles = self.fc_quantiles(outputs)
      quantiles = quantiles.view(-1, 1, self.num_quantiles)  # Reshape to [batch_size, num_actions, num_quantiles]

      # Dueling Architecture: Compute advantage
      advantage = self.fc_advantage(outputs)
      advantage = advantage.view(-1, self.n_actions, self.num_quantiles) # Reshape to [batch_size, num_actions, 1]

      # Compute Final Q-values
      q_values = quantiles + advantage - quantiles.mean(dim=1, keepdim=True)

      return F.softmax(q_values, dim=-1), None

    def __deepcopy__(self, memodict={}):
        # Target Network: Create a new instance of the class
        new_instance = QuantumRainbow(n_qubits = self.n_qubits,
                                      n_actions = self.n_actions,
                                      n_layers = self.n_layers,
                                      w_input = self.w_input,
                                      w_output = self.w_output,
                                      data_reupload = self.data_reupload,
                                      num_quantiles = self.num_quantiles)

        # Copy the fully connected layers for quantiles and advantage
        new_instance.fc_quantiles = copy.deepcopy(self.fc_quantiles, memodict)
        new_instance.fc_advantage = copy.deepcopy(self.fc_advantage, memodict)

        # Assign the quantum parts after copying
        new_instance.q_layers = copy.deepcopy(self.q_layers, memodict)
        new_instance.entropy = copy.deepcopy(self.entropy, memodict)

        return new_instance


env = gym.make(args.task)

train_envs = ts.env.DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

# Use your defined network
state_shape = env.observation_space.shape[0]  # equivalent to 4 for CartPole-v1
action_shape = env.action_space.n  # equivalent to 2 for CartPole-v1

net = QuantumRainbow(n_qubits=state_shape, n_actions=action_shape, n_layers=args.qnn_layers, w_input=True, w_output=True, 
                     data_reupload=args.data_reupload, num_quantiles=args.num_quantiles)
# net = net.to(args.device)
optim = torch.optim.RMSprop(net.parameters(), lr=args.lr)
policy = RainbowPolicy(net, optim, discount_factor=args.gamma, num_atoms=args.num_quantiles,
                       v_min = args.v_min, v_max = args.v_max,
                       estimation_step=args.n_step,
                       target_update_freq=args.target_update_freq)
# policy = policy.to(args.device)

buffer = PrioritizedVectorReplayBuffer(alpha=args.alpha, beta=args.beta, total_size=args.buffer_size, buffer_num=args.training_num)  # max size of the replay buffer
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)


from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
writer = SummaryWriter(f'log_{args.log_num}/PennyLane_{args.task}_Quantum_Rainbow')
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
    episode_per_test=1000,  # number of episodes per test
    batch_size=args.batch_size,  # batch size for updating model
    train_fn=lambda epoch, env_step: policy.set_eps(args.eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(args.eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    logger=logger)

print(f'Finished training! Use {result["duration"]}')


path = f'/scratch/connectome/justin/log_{args.log_num}/PennyLane_{args.task}_Quantum_Rainbow_{args.log_num}.pth'
torch.save(policy.state_dict(), path)

policy.load_state_dict(torch.load(path))
policy.eval()
result = test_collector.collect(n_episode=100, render=False)
print("Quantum Rainbow Final reward: {}, length: {}".format(result["rews"].mean(), result["lens"].mean()))


path = f'/scratch/connectome/justin/log_{args.log_num}/PennyLane_{args.task}_Quantum_Rainbow_{args.log_num}.hdf5'
ts.data.ReplayBuffer.save_hdf5(buffer, path)

load_buffer= ts.data.ReplayBuffer.load_hdf5(path)

buffer_obs = buffer.get(index=10, key='obs', stack_num=args.buffer_size)
buffer_obs_next = buffer.get(index=10, key='obs_next', stack_num=args.buffer_size)
buffer_rew = buffer.get(index=10, key='rew', stack_num=args.buffer_size)
buffer_act = buffer.get(index=10, key='act', stack_num=args.buffer_size)

buffer_df = pd.DataFrame(buffer_obs, columns=['Cart_Position', 'Cart_Velocity', 'Pole_Angle', 'Pole_Angular_Velocity'])
buffer_df2 = pd.DataFrame(buffer_obs_next, columns=['Cart_Position2', 'Cart_Velocity2', 'Pole_Angle2', 'Pole_Angular_Velocity2'])
buffer_df3 = pd.DataFrame(buffer_rew, columns=['Reward'])
buffer_df4 = pd.DataFrame(buffer_act, columns=['Action'])

buffer_res = pd.concat([buffer_df, buffer_df2], axis=1)
buffer_res = pd.concat([buffer_res, buffer_df3], axis=1)
buffer_res = pd.concat([buffer_res, buffer_df4], axis=1)

buffer_res.to_csv(f'/scratch/connectome/justin/log_{args.log_num}/PennyLane_{args.task}_Quantum_Rainbow_{args.log_num}.csv')

import re
import pandas as pd

numbers = [[float(re.search(r'\d+\.\d+', str(value)).group()) for value in sublist] for sublist in entropy_out]
df = pd.DataFrame(numbers)
df = df.transpose()
df.to_csv(f'/scratch/connectome/justin/log_{args.log_num}/PennyLane_{args.task}_Quantum_Rainbow_entropy_{args.log_num}.csv')
