# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-Layer Perceptron networks
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import numpy as np


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.
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
        Initialize MLP.
        
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
        
        layers = []
        prev_dim = input_dim
        
        activation_fn = self._get_activation(activation)
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(activation_fn)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "identity": nn.Identity(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LayerNormMLP(nn.Module):
    """
    MLP with layer normalization and optional residual connections.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: Optional[str] = None,
        use_residual: bool = False,
    ):
        super().__init__()
        
        self.use_residual = use_residual and len(hidden_dims) > 0
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0] if hidden_dims else output_dim)
        self.input_norm = nn.LayerNorm(hidden_dims[0] if hidden_dims else output_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        
        prev_dim = hidden_dims[0] if hidden_dims else input_dim
        
        for hidden_dim in hidden_dims[1:]:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.hidden_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim) if hidden_dims else None
        
        # Activations
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation) if output_activation else None
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        out = self.input_layer(x)
        out = self.input_norm(out)
        out = self.activation(out)
        
        # Hidden layers with optional residual
        for layer, norm in zip(self.hidden_layers, self.hidden_norms):
            residual = out
            out = layer(out)
            out = norm(out)
            out = self.activation(out)
            
            if self.use_residual and out.shape == residual.shape:
                out = out + residual
        
        # Output layer
        if self.output_layer is not None:
            out = self.output_layer(out)
        
        if self.output_activation is not None:
            out = self.output_activation(out)
        
        return out


class EnsembleMLP(nn.Module):
    """
    Ensemble of MLPs for uncertainty estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int = 5,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        
        self.num_ensemble = num_ensemble
        self.output_dim = output_dim
        
        # Create ensemble members
        self.ensemble = nn.ModuleList([
            MLP(input_dim, output_dim, hidden_dims, activation)
            for _ in range(num_ensemble)
        ])
    
    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            return_all: If True, return all ensemble outputs
            
        Returns:
            If return_all: [num_ensemble, batch_size, output_dim]
            Else: mean of ensemble outputs [batch_size, output_dim]
        """
        outputs = torch.stack([member(x) for member in self.ensemble], dim=0)
        
        if return_all:
            return outputs
        else:
            return outputs.mean(dim=0)
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Get uncertainty as standard deviation across ensemble."""
        outputs = self.forward(x, return_all=True)
        return outputs.std(dim=0)


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = [256, 256],
    activation: str = "relu",
    output_activation: Optional[str] = None,
    **kwargs
) -> MLP:
    """Factory function to create MLP."""
    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
        **kwargs
    )
