# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: How to register a task with custom SAC configuration

Add this to your task's __init__.py file to enable SAC training.
"""

# Example task registration with SAC support:

# import gymnasium as gym
# from . import agents
#
# gym.register(
#     id="Isaac-YourTask-Direct-v0",
#     entry_point=f"{__name__}.your_env:YourEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.your_env:YourEnvCfg",
#         # Other RL library configs...
#         "skrl_sac_cfg_entry_point": f"{agents.__name__}:sac_config.yaml",
#     },
# )


# Example sac_config.yaml content:
EXAMPLE_CONFIG = """
# SAC Configuration for IsaacLab Task
seed: 42

models:
  separate: True
  
  policy:
    class: GaussianMixin
    clip_actions: False
    clip_log_std: True
    min_log_std: -20.0
    max_log_std: 2.0
    initial_log_std: 0.0
    network:
      - name: net
        input: STATES
        layers: [256, 256]
        activations: [relu, relu]
    output: ACTIONS
    
  critic_1:
    class: DeterministicMixin
    clip_actions: False
    network:
      - name: net
        input: concatenate([STATES, ACTIONS])
        layers: [256, 256]
        activations: [relu, relu]
    output: ONE
    
  critic_2:
    class: DeterministicMixin
    clip_actions: False
    network:
      - name: net
        input: concatenate([STATES, ACTIONS])
        layers: [256, 256]
        activations: [relu, relu]
    output: ONE
    
  target_critic_1:
    class: DeterministicMixin
    clip_actions: False
    network:
      - name: net
        input: concatenate([STATES, ACTIONS])
        layers: [256, 256]
        activations: [relu, relu]
    output: ONE
    
  target_critic_2:
    class: DeterministicMixin
    clip_actions: False
    network:
      - name: net
        input: concatenate([STATES, ACTIONS])
        layers: [256, 256]
        activations: [relu, relu]
    output: ONE

memory:
  class: RandomMemory
  memory_size: 1000000

agent:
  class: SAC
  gradient_steps: 1
  batch_size: 256
  discount_factor: 0.99
  polyak: 0.005
  actor_learning_rate: 3.0e-4
  critic_learning_rate: 3.0e-4
  learning_rate_scheduler: null
  learning_rate_scheduler_kwargs: {}
  random_timesteps: 0
  learning_starts: 10000
  grad_norm_clip: 1.0
  auto_entropy_tuning: true
  initial_entropy_value: 0.2
  target_entropy: null
  state_preprocessor: null
  state_preprocessor_kwargs: null
  clip_actions: true
  exploration_noise: 0.0
  experiment:
    directory: "your_task_sac"
    experiment_name: ""
    write_interval: auto
    checkpoint_interval: auto

trainer:
  class: SequentialTrainer
  timesteps: 1000000
  environment_info: log
"""

# How to train:
# 
# Using the custom train.py:
# python /path/to/sac_isaaclab/train.py --task Isaac-YourTask-Direct-v0 --num_envs 4096 --headless
#
# Or with explicit hyperparameters:
# python /path/to/sac_isaaclab/train.py \\
#     --task Isaac-YourTask-Direct-v0 \\
#     --num_envs 4096 \\
#     --batch_size 512 \\
#     --learning_rate 1e-4 \\
#     --buffer_size 2000000 \\
#     --warmup_steps 20000 \\
#     --headless
