# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Storage package
"""

from .replay_buffer import ReplayBuffer
from .prioritized_buffer import PrioritizedReplayBuffer, RankBasedPrioritizedBuffer
from .storage_utils import (
    RolloutBuffer,
    FrameStack,
    RewardNormalizer,
    StateNormalizer,
    merge_batches,
    split_batch,
)


def make_replay_buffer(
    buffer_type: str = "random",
    buffer_size: int = 1000000,
    obs = None,
    action_shape = None,
    device: str = "cuda:0",
    **kwargs
):
    """
    Factory function to create replay buffer.
    
    Args:
        buffer_type: "random" or "priority" (default: "random")
        buffer_size: Maximum number of transitions
        obs: Shape of observations
        action_shape: Shape of actions
        device: Device for tensors
        **kwargs: Additional arguments for specific buffer types
        
    Returns:
        ReplayBuffer instance
        
    Example:
        >>> buffer = make_replay_buffer("random", buffer_size=1000000, obs=(8,), action_shape=(2,))
        >>> buffer = make_replay_buffer("priority", buffer_size=1000000, obs=(8,), action_shape=(2,))
    """
    buffer_type = buffer_type.lower()
    
    if buffer_type == "random":
        return ReplayBuffer(
            buffer_size=buffer_size,
            obs=obs,
            action_shape=action_shape,
            device=device,
        )
    elif buffer_type in ["priority", "per", "prioritized"]:
        return PrioritizedReplayBuffer(
            buffer_size=buffer_size,
            obs=obs,
            action_shape=action_shape,
            device=device,
            alpha=kwargs.get("alpha", 0.6),
            beta=kwargs.get("beta", 0.4),
            beta_increment=kwargs.get("beta_increment", 0.001),
            epsilon=kwargs.get("epsilon", 1e-6),
            gpu_mem_log_path=kwargs.get("gpu_mem_log_path", None),
        )
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}. Choose 'random' or 'priority'")


__all__ = [
    "ReplayBuffer",
    "NStepReplayBuffer",
    "PrioritizedReplayBuffer",
    "RankBasedPrioritizedBuffer",
    "RolloutBuffer",
    "FrameStack",
    "RewardNormalizer",
    "StateNormalizer",
    "merge_batches",
    "split_batch",
    "make_replay_buffer",
]
