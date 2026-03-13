# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Base network classes for SAC
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable


class BaseNetwork(nn.Module):
    """
    Base network class providing common functionality.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: Optional[str] = None,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initialize base network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            output_activation: Output activation function name
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Output activation
        self.output_activation = self._get_activation(output_activation) if output_activation else None
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.network(x)
        output = self.output_layer(features)
        
        if self.output_activation is not None:
            output = self.output_activation(output)
        
        return output


class MLP(BaseNetwork):
    """
    Multi-Layer Perceptron.
    Simple feedforward network.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=output_activation,
            **kwargs
        )


class ResidualBlock(nn.Module):
    """
    Residual block for building deeper networks.
    """
    
    def __init__(self, dim: int, activation: str = "relu", use_layer_norm: bool = True):
        super().__init__()
        
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = self._get_activation(activation)
        
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x
        
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.fc2(out)
        out = self.norm2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class ResidualNetwork(nn.Module):
    """
    Network with residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = self._get_activation(activation)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation, use_layer_norm)
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.activation(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_layer(x)


class FeatureExtractor(nn.Module):
    """
    Feature extraction network that can be shared between policy and value.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
