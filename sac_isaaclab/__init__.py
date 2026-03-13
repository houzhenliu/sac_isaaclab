# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
SAC (Soft Actor-Critic) Implementation for IsaacLab
Standalone version - no skrl dependencies except environment wrapping

Example:
    >>> from sac_isaaclab import SAC, GaussianActor, TwinQCritic
    >>> from sac_isaaclab.storage import ReplayBuffer
    >>> 
    >>> # Create networks
    >>> actor = GaussianActor(obs_dim=8, action_dim=2)
    >>> critic = TwinQCritic(obs_dim=8, action_dim=2)
    >>> target_critic = TwinQCritic(obs_dim=8, action_dim=2)
    >>> 
    >>> # Create agent
    >>> agent = SAC(actor, critic, target_critic, observation_space, action_space)
"""

# Version
__version__ = "0.1.0"

# Main exports - standalone version (no skrl dependencies)
from .modules import SAC, SAC_DEFAULT_CONFIG
from .network import GaussianActor, TwinQCritic, TwinQNetwork
from .storage import ReplayBuffer, PrioritizedReplayBuffer, make_replay_buffer

__all__ = [
    "SAC",
    "SAC_DEFAULT_CONFIG",
    "GaussianActor",
    "TwinQCritic",
    "TwinQNetwork",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "make_replay_buffer",
]
