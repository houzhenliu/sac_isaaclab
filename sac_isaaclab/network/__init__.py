# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Network package for SAC
"""

from .base import BaseNetwork, MLP, ResidualBlock, ResidualNetwork, FeatureExtractor
from .mlp import MLP as SimpleMLP, LayerNormMLP, EnsembleMLP, create_mlp
from .actor import GaussianActor, SimpleActor, SharedActorCritic
from .critic import QCritic, StateCritic, TwinQCritic, EnsembleCritic, FeatureCritic

# Aliases for backward compatibility
TwinQNetwork = TwinQCritic

__all__ = [
    # Base
    "BaseNetwork",
    "ResidualBlock",
    "ResidualNetwork",
    "FeatureExtractor",
    # MLP
    "SimpleMLP",
    "LayerNormMLP",
    "EnsembleMLP",
    "create_mlp",
    # Actor
    "GaussianActor",
    "SimpleActor",
    "SharedActorCritic",
    # Critic
    "QCritic",
    "StateCritic",
    "TwinQCritic",
    "TwinQNetwork",  # Alias
    "EnsembleCritic",
    "FeatureCritic",
]
