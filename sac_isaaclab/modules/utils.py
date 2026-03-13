# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions for SAC implementation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """
    Soft update target network parameters.
    
    Args:
        target: Target network
        source: Source network
        tau: Interpolation factor (0 < tau <= 1)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Hard update target network parameters."""
    target.load_state_dict(source.state_dict())


def init_weights(module: torch.nn.Module, init_type: str = "xavier") -> None:
    """
    Initialize network weights.
    
    Args:
        module: Network module
        init_type: Initialization type ('xavier', 'orthogonal', 'normal')
    """
    if init_type == "xavier":
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    elif init_type == "orthogonal":
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    elif init_type == "normal":
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def get_activation(activation_name: str):
    """Get activation function by name."""
    activations = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
        "sigmoid": torch.nn.Sigmoid,
        "silu": torch.nn.SiLU,
        "gelu": torch.nn.GELU,
    }
    return activations.get(activation_name.lower(), torch.nn.ReLU)


def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    return 0.0


def compute_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Compute entropy of Gaussian distribution."""
    import math
    return 0.5 * (log_std.shape[-1] * (1.0 + math.log(2 * math.pi)) + log_std.sum(dim=-1))


def normalize_observations(observations, running_mean, running_var, epsilon=1e-8):
    """Normalize observations using running statistics."""
    return (observations - running_mean) / torch.sqrt(running_var + epsilon)


class RunningMeanStd:
    """Running mean and standard deviation calculator."""
    
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        """Update running statistics with new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from precomputed moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x, epsilon=1e-8):
        """Normalize using running statistics."""
        return (x - self.mean) / np.sqrt(self.var + epsilon)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_timestep(timestep: int) -> str:
    """Format timestep for logging."""
    if timestep >= 1e6:
        return f"{timestep / 1e6:.2f}M"
    elif timestep >= 1e3:
        return f"{timestep / 1e3:.2f}K"
    else:
        return str(timestep)
