# A more Effective SAC Implementation for IsaacLab
🚀 A much more efficient & flexible SAC implementation for IsaacLab

✅ Larger Replay Buffer — richer experience distribution & stronger stability


♻ Optimized Off-Policy Training — higher sample efficiency in IsaacLab

## Requirements

| Package | Version |
|---------|---------|
| Python | 3.11 (tested with 3.11.13) |
| PyTorch | >=2.7.0 (tested with 2.7.0+cu128) |
| NumPy | <2.0 (tested with 1.26.0) |
| IsaacLab | >= 2.3.0 |
| IsaacSim |  5.1.0 |
| platform | Linux-64(preferred), Windows-64 |
| docker | >=29.0.0 (tested with 29.2.0) |
## Quick Start
### 1. IsaacLab Setup
Follow [IsaacLab 2.3.0 Docker Installation](https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/setup/installation/index.html) to download IsaacLab and set up the Docker environment. Ensure you have NVIDIA drivers and Docker installed.

### 2. Enter the IsaacLab Container

```bash
cd path/to/IsSacLab

docker/container.py start 
#if you haven't started the container yet

docker/container.py enter base
```

### 3. Install SAC-PER
Copy the resonsitory into the container (optional, for easier access)
```bash
git clone github.com/houzhenliu/sac-isaaclab

./isaaclab.sh -p -m pip install -e sac-isaaclab
```

### 4. Add configuration to IsaacLab source code
For example, if you want to run our algorithm using task Isaac-Ant-Direct-v0, you can modify `IsaacLab/source/isaaclab-tasks/direct/ant/__init__.py` as follows:

```diff
gym.register(
    id="Isaac-Ant-Direct-v0",
    entry_point=f"{__name__}.ant_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env:AntEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
+       "sac_cfg_entry_point": f"{agents.__name__}:sac_cfg.yaml",
    },
)
```

Then you can copy sac-isaaclab/sac_cfg.yaml to `IsaacLab/source/isaaclab-tasks/direct/ant/agent` directory.

### 5. Start Training
```bash
# Run the training script with default configuration
./isaaclab.sh -p sac-isaaclab/train.py --task Isaac-Ant-Direct-v0 --num_envs 256

# Run the training script with video recording and headless mode
./isaaclab.sh -p sac-isaaclab/train.py --task Isaac-Ant-Direct-v0 --num_envs 256 --video --headless

# Run the training script with PER enabled (also you can modify sac_cfg.yaml to enable PER by default)
./isaaclab.sh -p sac-isaaclab/train.py --task Isaac-Ant-Direct-v0 --num_envs 256 --use_per
```



