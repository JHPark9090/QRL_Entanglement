# General imports
import numpy as np
import pandas as pd
import copy
import os
import random
import pickle
import functools
# from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.special import kl_div

# PennyLane
import pennylane as qml

# PyTorch imports
import torch


def log_negativity(states):
    """Log Negativity
    Input: Quantum States"""
    log_neg = []
    for i in range(len(states)):
        log_neg_i = []
        for j in range(len(states[i])):
            dm = qml.math.dm_from_state_vector(states[i][j]).cpu().detach().numpy()   # Obtain density matrix from quantum state vector
            # Partial transpose of density matrix with respect to subsystem A (qubit 0,1)
            for k in range(8):
                for l in range(8,16):
                    dm[[k, l], :] = dm[[l, k], :]
            for k in range(8):
                for l in range(8,16):
                    dm[:, [k, l]] = dm[:, [l, k]]
            # Trace norm
            trace_norm = np.linalg.norm(dm, ord='nuc')
            # Log Negativity of the state
            log_neg_i.append(np.log2(trace_norm))
        log_neg.append(log_neg_i)
    return log_neg


def coherent_info(states):
    """Coherent Information
    Input: Quantum States"""
    co_info = []
    for i in range(len(states)):
        co_info_i = []
        for j in range(len(states[i])):
            dm = qml.math.dm_from_state_vector(states[i][j]).cpu()   # Obtain density matrix from quantum state vector
            rd_dm = qml.math.reduce_dm(dm, indices=[0,1])   # Obtain reduced density matrix of the quantum state vector (i.e., the density matrix of the subsystem A)
            co_info_i.append(qml.math.vn_entropy(rd_dm, indices=[0])-qml.math.vn_entropy(dm, indices=[0]))
        co_info.append(co_info_i)
    return co_info


def entangle_cap(states, n_qubits):
    """Entangling Capability"""    
    entangling_capability = []
    for i in range(len(states)):
        Q = []  # Meyer–Wallach entanglement measure
        
        for j in range(len(states[i])):      # Calculate Meyer–Wallach entanglement measure for each quantum state
            state_vector = states[i][j].cpu()
            entropy = 0
            if isinstance(state_vector, torch.Tensor) and state_vector.requires_grad:
                state_vector = state_vector.detach().numpy()
            elif isinstance(state_vector, torch.Tensor):
                state_vector = state_vector.numpy()
            for k in range(n_qubits):
                dens = qml.math.reduce_dm(qml.math.dm_from_state_vector(state_vector), indices=[k])
                trace = np.trace(np.linalg.matrix_power(dens, 2))
                entropy += 0.5*(trace.real)
            Q.append(4*entropy/n_qubits)
        
        ent_cap = np.sum(Q)/len(states[i])    # Calculate Entangling Capability as the average Meyer–Wallach entanglement measure
        entangling_capability.append(ent_cap)    
    
    return entangling_capability


def prob_haar(num_samples, num_qubits) -> np.ndarray:
    """Returns probability density function of fidelities for Haar Random States"""
    fidelity = np.linspace(0, 1, num_samples)
    return (2**num_qubits - 1) * (1 - fidelity + 1e-8) ** (2**num_qubits - 2)
    
def expressivity(states, num_qubits):
    express_out = []
    # Calculate fidelities from quantum states (generated from the Quantum RL model)    
    for i in range(len(states)):
        express = []
        for j in range(len(states[i])):
            # Calculate fidelities of quantum states derived from quantum circuits
            fidelity = []
            for k in range(len(states[i])):
                fid = qml.math.fidelity(qml.math.dm_from_state_vector(states[i][j]), qml.math.dm_from_state_vector(states[i][k])).cpu()
                if isinstance(fid, torch.Tensor) and fid.requires_grad:
                    fid = fid.detach().numpy()
                elif isinstance(fid, torch.Tensor):
                    fid = fid.numpy()
                fidelity.append(fid)
            # Haar_prob: Obtain Haar distribution
            haar = prob_haar(len(states[i]), num_qubits)
            haar_prob: np.ndarray = haar / float(haar.sum())
            # PQC_prob: Build histogram of distribution to estimate PQC probability distribution of fidelities
            bin_edges: np.ndarray
            pqc_hist, bin_edges = np.histogram(
                np.array(fidelity), len(states[i]), range=(0, 1), density=True
            )
            pqc_prob: np.ndarray = pqc_hist / float(pqc_hist.sum())
            #  Kullback-Leibler Divergence of PQC_prob and Haar_prob
            express_i = np.sum(kl_div(pqc_prob, haar_prob))
            express.append(express_i)
        express_out.append(express)

#     if plot == True:
#         fig = plt.figure(figsize=(9.6, 6))

#         plt.ylim([1e-3, 1])
#         plt.ylabel(r"$Expr. D_{KL}$")
#         plt.xlabel("Repetitions")

#         plt.yscale('log')
#         plt.plot(len(states), express_out, marker='o', ls='--')
#         plt.tight_layout()
#         plt.show()

    return express_out


def _torch_jac(circ):
    """Torch jacobian as a callable function"""
    import torch

    def wrapper(*args, **kwargs):
        loss = functools.partial(circ, **kwargs)
        if len(args) > 1:
            return torch.autograd.functional.jacobian(loss, args, create_graph=True)
        return torch.autograd.functional.jacobian(loss, *args, create_graph=True)

    return wrapper

def _compute_cfim(p, dp):
    r"""Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives
    I.e. it computes :math:`classical_fisher_{ij} = \sum_\ell (\partial_i p_\ell) (\partial_i p_\ell) / p_\ell`
    """
    # Exclude values where p=0 and calculate 1/p
    # p = torch.tensor(p.real, dtype=torch.float64)
    nonzeros_p = qml.math.where(p > 0, p, qml.math.ones_like(p))
    one_over_p = qml.math.where(p > 0, qml.math.ones_like(p), qml.math.zeros_like(p))
    one_over_p = one_over_p / nonzeros_p
    # Multiply dp and p
    # Note that casting and being careful about dtypes is necessary as interfaces
    # typically treat derivatives (dp) with float32, while standard execution (p) comes in float64
    dp = qml.math.cast_like(dp, p)
    dp = qml.math.reshape(
        dp, (len(p), -1)
    )  # Squeeze does not work, as you could have shape (num_probs, num_params) with num_params = 1
    dp_over_p = qml.math.transpose(dp) * one_over_p  # creates (n_params, n_probs) array

    # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)
    return dp_over_p @ dp

def effective_dimension(circuit, input_params, n_qubits, n_layers, n):
    # d = len(list(circuit.parameters()))
    d = len(input_params)
    # d = n_qubits+2*n_qubits*n_layers
    j = _torch_jac(circuit)(input_params)
    if isinstance(j, tuple):
        res = []
        for j_i in j:
            res.append(_compute_cfim(input_params, j_i))
        if len(j) == 1:
            return res[0]

        return res
    fisher = _compute_cfim(circuit(input_params), j).detach().numpy()
    fisher_trace = np.trace(fisher)
#     avg_fisher = np.average(np.reshape(fisher, (len(input_params), n_qubits, d, d)), axis=1)
#     normalized_fisher = d * avg_fisher / fisher_trace
    normalized_fisher = d * fisher / fisher_trace

    dataset_size = n
    if not isinstance(dataset_size, int) and len(dataset_size) > 1:
        # expand dims for broadcasting
        normalized_fisher = np.expand_dims(normalized_fisher, axis=0)
        n_expanded = np.expand_dims(np.asarray(dataset_size), axis=(1, 2, 3))
        logsum_axis = 1
    else:
        n_expanded = np.asarray(dataset_size)
        logsum_axis = None

    # calculate effective dimension for each data sample size out
    # of normalized normalized_fisher
    f_mod = normalized_fisher * n_expanded / (2 * np.pi * np.log(n_expanded))
    one_plus_fmod = np.eye(d) + f_mod
    # take log. of the determinant because of overflow
    dets = np.linalg.slogdet(one_plus_fmod)[1]
    # divide by 2 because of square root
    dets_div = dets / 2
    effective_dims = (2
                      * (logsumexp(dets_div, axis=logsum_axis) - np.log(len(input_params)))
                      / np.log(dataset_size / (2 * np.pi * np.log(dataset_size)))
        )

    return np.squeeze(effective_dims)




