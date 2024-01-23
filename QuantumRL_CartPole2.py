# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
import random
import pickle
from collections import namedtuple
from datetime import datetime
import functools
# from scipy.optimize import minimize

# PennyLane
import pennylane as qml

from QuantumCircuitMetrics import log_negativity, coherent_info, entangle_cap, expressivity, effective_dimension
# from QuantumModels import QuantumDQN, QuantumQRDQN, QuantumRainbow

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# from tqdm.autonotebook import tqdm

# OpenAI Gym import
import gymnasium as gym

# from distributed import init_distributed

# Fix seed for reproducibility
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed);

# To get smooth animations on Jupyter Notebooks.
# Note: these plotting function are taken from https://github.com/ageron/handson-ml2

import tianshou as ts
from tianshou.policy import DQNPolicy, QRDQNPolicy, RainbowPolicy
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
# from tianshou_trainer_modified import OffpolicyTrainer
from tianshou.utils.net.discrete import NoisyLinear

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='CartPole-v1')
    parser.add_argument("--model", type=str, default='Quantum_DQN')    
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
    parser.add_argument("--resume", action="store_false", default=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()

args=get_args()


def encode(n_qubits, inputs):
    """Encoding Layer"""
    for wire in range(n_qubits):
        qml.RX(inputs[wire], wires=wire)

def layer(n_qubits, y_weight, z_weight):
    """Parametrized Quantum Layers"""    
    for wire, y_weight in enumerate(y_weight):
        qml.RY(y_weight, wires=wire)
    for wire, z_weight in enumerate(z_weight):
        qml.RZ(z_weight, wires=wire)
    for wire in range(n_qubits):
        qml.CZ(wires=[wire, (wire + 1) % n_qubits])

def measure(n_qubits, n_output):
    """Measurement Layer"""
    if n_output==2:
        return [
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))]
    elif n_output==4:
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]


def get_model(n_qubits, n_layers, n_output, data_reupload, return_val=True, return_prob=False):
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
        if return_val:    # Expectation Value 
            return measure(n_qubits, n_output)
        elif return_prob:      # Probability of each computational basis state
            return qml.probs(wires=range(n_qubits))
        else:
            return qml.state()      # Quantum State

    if (return_val==True) or return_prob:
        model = qml.qnn.TorchLayer(circuit, shapes)
    else:
        model = circuit  
    
    return model
    

runtime=0 


class QuantumDQN(nn.Module):
    def __init__(self, n_qubits, n_actions, n_layers, w_input, w_output, data_reupload, device, path):
        super(QuantumDQN, self).__init__()
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.w_input = w_input
        self.w_output = w_output
        self.data_reupload = data_reupload
        self.device = device
        self.path = path
        self.q_layers = get_model(n_qubits=n_qubits,
                                  n_layers=n_layers,
                                  n_output=n_actions,
                                  data_reupload=data_reupload,
                                  return_val=True, return_prob=False)  # Set return_val=True to get the model outputs
        
        self.probs = get_model(n_qubits=n_qubits,
                               n_layers=n_layers,
                               n_output=n_actions,
                               data_reupload=data_reupload,
                               return_val=False, return_prob=True)  # Set return_prob=True to get the probability distribution of all computational basis elements
        
        self.states = get_model(n_qubits=n_qubits,
                                n_layers=n_layers,
                                n_output=n_actions,
                                data_reupload=data_reupload,
                                return_val=False, return_prob=False)  # Set return_val=False, return_prob=False to get the quantum states
        
        if w_input:
            self.w_input2 = Parameter(torch.Tensor(self.n_qubits))
            nn.init.normal_(self.w_input2)
        else:
            self.register_parameter('w_input', None)
        if w_output:
            self.w_output2 = Parameter(torch.Tensor(self.n_actions))
            nn.init.normal_(self.w_output2, mean=90.)
        else:
            self.register_parameter('w_output', None)

    def forward(self, inputs, **kwargs):   
        batch_size = inputs.shape[0]  # Get the batch size
        outputs = []
        ED = []  # Effective Dimension
        state = []  # Quantum states
        
        global runtime
        
        for i in range(batch_size):
            input_i = inputs[i]  # Get the i-th input in the batch
            if not isinstance(input_i, torch.Tensor):
                input_i = torch.tensor(input_i).to(self.device)
#             input_i = torch.tensor(input_i, dtype=torch.complex64)  # Convert input_i to a PyTorch tensor
            if self.w_input2 is not None:
                if not isinstance(self.w_input2, torch.Tensor):
                    self.w_input2 = torch.tensor(self.w_input2, dtype=torch.complex64).to(self.device)
                input_i = input_i * self.w_input2
            input_i = torch.atan(input_i)
            output_i = self.q_layers(input_i)            
            output_i = (1 + output_i) / 2
            outputs.append(output_i)
            
            self.probs.load_state_dict(self.q_layers.state_dict())
            ED_i = effective_dimension(self.probs, input_i.cpu(), self.n_qubits, self.n_layers, 5000)     # Effective Dimension
            ED.append(ED_i)
            state_i = self.states(input_i, self.q_layers.state_dict()['y_weights'], self.q_layers.state_dict()['z_weights'])     # Quantum States
            state.append(state_i)

        outputs = torch.stack(outputs)  # Stack outputs along the batch dimension
        runtime += 1 

        ED_out.append(ED)
        state_out.append(state)
        torch.save({"ED_out": ED_out, "state_out": state_out}, os.path.join(self.path, f"metric_checkpoint_{runtime}.pth"))

        if self.w_output2 is not None:
            if not isinstance(self.w_output2, torch.Tensor):
                self.w_output2 = torch.tensor(self.w_output2, dtype=torch.complex64).to(self.device)           
            outputs = outputs * self.w_output2
        else:
            outputs = 90 * outputs
            outputs = outputs.view(-1, self.n_qubits * 2)
            
        return outputs, None


class QuantumQRDQN(nn.Module):
    def __init__(self, n_qubits, n_actions, n_layers, w_input, w_output, data_reupload, device, num_quantiles, neurons, path):
        super(QuantumQRDQN, self).__init__()
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.w_input = w_input
        self.w_output = w_output
        self.data_reupload = data_reupload
        self.device = device
        self.num_quantiles=num_quantiles
        self.neurons = neurons
        self.path = path
        self.q_layers = get_model(n_qubits=n_qubits,
                                  n_layers=n_layers,
                                  n_output=n_actions,
                                  data_reupload=data_reupload,
                                  return_val=True, return_prob=False)  # Set return_val=True to get the model outputs
        
        self.probs = get_model(n_qubits=n_qubits,
                               n_layers=n_layers,
                               n_output=n_actions,
                               data_reupload=data_reupload,
                               return_val=False, return_prob=True)  # Set return_prob=True to get the probability distribution of all computational basis elements
        
        self.states = get_model(n_qubits=n_qubits,
                                n_layers=n_layers,
                                n_output=n_actions,
                                data_reupload=data_reupload,
                                return_val=False, return_prob=False)  # Set return_val=False, return_prob=False to get the quantum states        
        
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
            nn.Linear(np.prod(self.n_actions), self.neurons), nn.BatchNorm1d(self.neurons), nn.ReLU(),
            nn.Linear(self.neurons, self.neurons), nn.BatchNorm1d(self.neurons), nn.ReLU(),
            nn.Linear(self.neurons, self.neurons), nn.BatchNorm1d(self.neurons), nn.ReLU(),
            nn.Linear(self.neurons, np.prod(self.n_actions) * num_quantiles)
            )


    def forward(self, inputs, **kwargs):   
        batch_size = inputs.shape[0]  # Get the batch size
        outputs = []
        ED = []  # Effective Dimension
        state = []  # Quantum states
        
        global runtime
        
        for i in range(batch_size):
            input_i = inputs[i]  # Get the i-th input in the batch
            if not isinstance(input_i, torch.Tensor):
                input_i = torch.tensor(input_i).to(self.device)
#             input_i = torch.tensor(input_i, dtype=torch.complex64)  # Convert input_i to a PyTorch tensor
            if self.w_input2 is not None:
                if not isinstance(self.w_input2, torch.Tensor):
                    self.w_input2 = torch.tensor(self.w_input2).to(self.device)
                input_i = input_i * self.w_input2
            input_i = torch.atan(input_i)
            output_i = self.q_layers(input_i)            
            output_i = (1 + output_i) / 2
            outputs.append(output_i)
            
            self.probs.load_state_dict(self.q_layers.state_dict())
            ED_i = effective_dimension(self.probs, input_i.cpu(), self.n_qubits, self.n_layers, 5000)     # Effective Dimension
            ED.append(ED_i)
            state_i = self.states(input_i, self.q_layers.state_dict()['y_weights'], self.q_layers.state_dict()['z_weights'])     # Quantum States
            state.append(state_i)

        outputs = torch.stack(outputs)  # Stack outputs along the batch dimension
        runtime += 1 
        
        ED_out.append(ED)
        state_out.append(state)
        torch.save({"ED_out": ED_out, "state_out": state_out}, os.path.join(self.path, f"metric_checkpoint_{runtime}.pth"))

        if self.w_output2 is not None:
            if not isinstance(self.w_output2, torch.Tensor):
                self.w_output2 = torch.tensor(self.w_output2).to(self.device)           
            outputs = outputs * self.w_output2
        else:
            outputs = 90 * outputs
            outputs = outputs.view(-1, self.n_qubits * 2)
       
        logits = self.fc_quantiles(outputs.to(torch.float32))
        return logits.view(batch_size, -1, self.num_quantiles), None

    
class QuantumRainbow(nn.Module):
    def __init__(self, n_qubits, n_actions, n_layers, w_input, w_output, data_reupload, device, num_quantiles, neurons, path):
        super(QuantumRainbow, self).__init__()
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.w_input = w_input
        self.w_output = w_output
        self.data_reupload = data_reupload
        self.device = device
        self.num_quantiles = num_quantiles
        self.neurons = neurons
        self.path = path
        self.q_layers = get_model(n_qubits=n_qubits,
                                  n_layers=n_layers,
                                  n_output=n_actions,
                                  data_reupload=data_reupload,
                                  return_val=True, return_prob=False)  # Set return_val=True to get the model outputs
        
        self.probs = get_model(n_qubits=n_qubits,
                               n_layers=n_layers,
                               n_output=n_actions,
                               data_reupload=data_reupload,
                               return_val=False, return_prob=True)  # Set return_prob=True to get the probability distribution of all computational basis elements
        
        self.states = get_model(n_qubits=n_qubits,
                                n_layers=n_layers,
                                n_output=n_actions,
                                data_reupload=data_reupload,
                                return_val=False, return_prob=False)  # Set return_val=False, return_prob=False to get the quantum states
        
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
            NoisyLinear(self.n_actions, self.neurons), nn.ReLU(),
            NoisyLinear(self.neurons, self.neurons), nn.ReLU(),
            NoisyLinear(self.neurons, num_quantiles)
        )
        self.fc_advantage = nn.Sequential(
            NoisyLinear(self.n_actions, self.neurons), nn.ReLU(),
            NoisyLinear(self.neurons, self.neurons), nn.ReLU(),
            NoisyLinear(self.neurons, self.n_actions*num_quantiles)
        )

    def forward(self, inputs, **kwargs):   
        batch_size = inputs.shape[0]  # Get the batch size
        outputs = []
        ED = []  # Effective Dimension
        state = []  # Quantum states
        global runtime
        
        for i in range(batch_size):
            input_i = inputs[i]  # Get the i-th input in the batch
            if not isinstance(input_i, torch.Tensor):
                input_i = torch.tensor(input_i).to(self.device)
#             input_i = torch.tensor(input_i, dtype=torch.complex64)  # Convert input_i to a PyTorch tensor
            if self.w_input2 is not None:
                if not isinstance(self.w_input2, torch.Tensor):
                    self.w_input2 = torch.tensor(self.w_input2).to(self.device)
                input_i = input_i * self.w_input2
            input_i = torch.atan(input_i)
            output_i = self.q_layers(input_i)            
            output_i = (1 + output_i) / 2
            outputs.append(output_i)
            
            self.probs.load_state_dict(self.q_layers.state_dict())
            ED_i = effective_dimension(self.probs, input_i.cpu(), self.n_qubits, self.n_layers, 5000)     # Effective Dimension
            ED.append(ED_i)
            state_i = self.states(input_i, self.q_layers.state_dict()['y_weights'], self.q_layers.state_dict()['z_weights'])     # Quantum States
            state.append(state_i)

        outputs = torch.stack(outputs)  # Stack outputs along the batch dimension
        runtime += 1 
        
        ED_out.append(ED)
        state_out.append(state)
        torch.save({"ED_out": ED_out, "state_out": state_out}, os.path.join(self.path, f"metric_checkpoint_{runtime}.pth"))

        if self.w_output2 is not None:
            if not isinstance(self.w_output2, torch.Tensor):
                self.w_output2 = torch.tensor(self.w_output2).to(self.device)           
            outputs = outputs * self.w_output2
        else:
            outputs = 90 * outputs
            outputs = outputs.view(-1, self.n_qubits * 2)

        # Distributional RL: Compute quantiles
        quantiles = self.fc_quantiles(outputs.to(torch.float32))
        quantiles = quantiles.view(-1, 1, self.num_quantiles)  # Reshape to [batch_size, num_actions, num_quantiles]

        # Dueling Architecture: Compute advantage
        advantage = self.fc_advantage(outputs.to(torch.float32))
        advantage = advantage.view(-1, self.n_actions, self.num_quantiles) # Reshape to [batch_size, num_actions, 1]

        # Compute Final Q-values
        q_values = quantiles + advantage - quantiles.mean(dim=1, keepdim=True)

        return F.softmax(q_values, dim=-1), None

        
log_path = f'log_{args.log_num}/PennyLane_{args.task}_{args.model}'
metric_checkpoint_path = f'{log_path}/metric_checkpoint'

# List all files in the directory
if args.resume:
    files = os.listdir(metric_checkpoint_path)
    # Initialize variables to store the maximum number and corresponding file name
    max_number = -1
    max_file = ""
    # Iterate through the files
    for file in files:
        if file.startswith("metric_checkpoint_") and file.endswith(".pth"):
            # Extract the number from the file name
            try:
                number = int(file.split("_")[2].split(".")[0])
                if number > max_number:
                    max_number = number
                    max_file = file
            except ValueError:
                continue
    maxfile_path = f"{metric_checkpoint_path}/{max_file}"

if args.resume:
    if os.path.exists(maxfile_path):
        checkpoint = torch.load(maxfile_path)
        ED_out = checkpoint["ED_out"]     # Load Effective Dimension
        state_out = checkpoint["state_out"]     # Load Quantum States
        print("Successfully restored Effective Dimension & Quantum States")
    else:
        ED_out = []  # Effective Dimension
        state_out = []  # Quantum States
        print("Failed to restore Effective Dimension & Quantum States")
else:
    ED_out = []  # Effective Dimension
    state_out = []  # Quantum States
    
env = gym.make(args.task)

train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

# Use the defined network
state_shape = env.observation_space.shape[0]  # equivalent to 4 for CartPole-v1
action_shape = env.action_space.n  # equivalent to 2 for CartPole-v1


if args.model == "Quantum_DQN":
    net = QuantumDQN(n_qubits=state_shape, n_actions=action_shape, n_layers=args.qnn_layers, w_input=True, w_output=True, 
                     data_reupload=args.data_reupload, device=args.device, path=metric_checkpoint_path)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(net, optim, discount_factor=args.gamma,
                       estimation_step=args.n_step,
                       target_update_freq=args.target_update_freq, is_double=False)
    policy = policy.to(args.device)
    buffer = VectorReplayBuffer(total_size=args.buffer_size, buffer_num=args.training_num)  # max size of the replay buffer
elif args.model == "Quantum_DDQN":
    net = QuantumDQN(n_qubits=state_shape, n_actions=action_shape, n_layers=args.qnn_layers, w_input=True, w_output=True, 
                     data_reupload=args.data_reupload, device=args.device, path=metric_checkpoint_path)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(net, optim, discount_factor=args.gamma,
                       estimation_step=args.n_step,
                       target_update_freq=args.target_update_freq, is_double=True)
    policy = policy.to(args.device)
    buffer = VectorReplayBuffer(total_size=args.buffer_size, buffer_num=args.training_num)  # max size of the replay buffer
elif args.model == "Quantum_QRDQN":
    net = QuantumQRDQN(n_qubits=state_shape, n_actions=action_shape, n_layers=args.qnn_layers, w_input=True, w_output=True, 
                       data_reupload=args.data_reupload, device=args.device, num_quantiles=args.num_quantiles, neurons=args.neurons, 
                       path=metric_checkpoint_path)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = QRDQNPolicy(net, optim, discount_factor=args.gamma, 
                     num_quantiles=args.num_quantiles,
                     estimation_step=args.n_step,
                     target_update_freq=args.target_update_freq, is_double=True)
    policy = policy.to(args.device)
    buffer = VectorReplayBuffer(total_size=args.buffer_size, buffer_num=args.training_num)  # max size of the replay buffer
elif args.model == "Quantum_Rainbow":
    net = QuantumRainbow(n_qubits=state_shape, n_actions=action_shape, n_layers=args.qnn_layers, w_input=True, w_output=True, 
                        data_reupload=args.data_reupload, device=args.device, num_quantiles=args.num_quantiles, neurons=args.neurons,
                        path=metric_checkpoint_path)
    net = net.to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = RainbowPolicy(net, optim, discount_factor=args.gamma, num_atoms=args.num_quantiles,
                       v_min = args.v_min, v_max = args.v_max,
                       estimation_step=args.n_step,
                       target_update_freq=args.target_update_freq)
    policy = policy.to(args.device)
    buffer = PrioritizedVectorReplayBuffer(alpha=args.alpha, beta=args.beta, total_size=args.buffer_size, buffer_num=args.training_num)  # max size of the replay buffer
    
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)


from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
# log_path = f'log_{args.log_num}/PennyLane_{args.task}_Quantum_DQN'
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)

def save_checkpoint_fn(epoch, env_step, gradient_step):
    # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    ckpt_path = os.path.join(log_path, "checkpoint.pth")
    # Example: saving by epoch num
    # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
    torch.save(
        {
            "model": policy.state_dict(),
            "optim": optim.state_dict(),
        },
            ckpt_path,
    )
    buffer_path = os.path.join(log_path, "train_buffer.pkl")
    with open(buffer_path, "wb") as f:
        pickle.dump(train_collector.buffer, f)
    return ckpt_path

if args.resume:
    # load from existing checkpoint
    print(f"Loading agent under {log_path}")
    ckpt_path = os.path.join(log_path, "checkpoint.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        policy.load_state_dict(checkpoint["model"])
        policy.optim.load_state_dict(checkpoint["optim"])
        print("Successfully restored policy and optim.")
    else:
        print("Failed to restore policy and optim.")
    buffer_path = os.path.join(log_path, "train_buffer.pkl")
    if os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            train_collector.buffer = pickle.load(f)
        print("Successfully restored buffer.")
    else:
        print("Failed to restore buffer.")
            
# Start training
result = OffpolicyTrainer(
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
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    resume_from_log = args.resume,
    save_checkpoint_fn=save_checkpoint_fn,
    logger=logger).run()

print(f'Finished training! Use {result["duration"]}')


path = f'log_{args.log_num}/PennyLane_{args.task}_{args.model}/PennyLane_{args.task}_{args.model}_{args.log_num}.pth'
torch.save(policy.state_dict(), path)

policy.load_state_dict(torch.load(path))
policy.eval()
result = test_collector.collect(n_episode=100, render=False)
print(args.model, " Final reward: {}, length: {}".format(result["rews"].mean(), result["lens"].mean()))


path = f'log_{args.log_num}/PennyLane_{args.task}_{args.model}/PennyLane_{args.task}_{args.model}_{args.log_num}.hdf5'
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

buffer_res.to_csv(f'log_{args.log_num}/PennyLane_{args.task}_{args.model}/PennyLane_{args.task}_{args.model}_{args.log_num}_buffer.csv')


def avg_per_iter(metric):
    ## Metrics: LN_out, CI_out, vn_entropy_out, ED_out
    mean = []
    for i in range(len(metric)):
        for j in range(len(metric[i])):
            x = metric[i][j]
            if isinstance(x, torch.Tensor) and x.requires_grad:
                x = x.detach().numpy()
            elif isinstance(x, torch.Tensor):
                x = x.numpy()
            metric[i][j] = x
        avg = np.mean(metric[i])
        mean.append(avg)
    return mean

LN_out = log_negativity(state_out)
CI_out = coherent_info(state_out)
EC = entangle_cap(state_out, state_shape)
express = expressivity(state_out, state_shape)

entangling_capability = pd.DataFrame(EC)

log_negativity = pd.DataFrame(avg_per_iter(LN_out))
coherent_info = pd.DataFrame(avg_per_iter(CI_out))
effective_dim = pd.DataFrame(avg_per_iter(ED_out))
expressibility = pd.DataFrame(avg_per_iter(express))

metrics = pd.concat([log_negativity, coherent_info, entangling_capability, effective_dim, expressibility], 
                    axis=1)
metrics.columns = ["Log_Negativity", "Coherent_Information", "Entangling_Capability",
                   "Effective_Dimension", "Expressibility"]

metrics.to_csv(f'log_{args.log_num}/PennyLane_{args.task}_{args.model}/PennyLane_{args.task}_Quantum_DQN_metrics_{args.log_num}.csv')
   
   
