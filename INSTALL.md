# Installation Guide

## Prerequisites

- Python 3.10 or 3.11 (tested with 3.11.13)
- CUDA 12.8 (for GPU support)
- IsaacLab installed and configured

## Environment Versions

This implementation is tested with the following package versions:

| Package | Version |
|---------|---------|
| Python | 3.11.13 |
| PyTorch | 2.7.0+cu128 |
| NumPy | 1.26.0 |
| Gymnasium | 1.2.1 |
| skrl | 1.4.3 |

## Installation Options

### Option 1: Install as a Package (Recommended)

```bash
cd sac_isaaclab

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Option 2: Install from Requirements

```bash
cd sac_isaaclab
pip install -r requirements.txt
```

### Option 3: Manual Installation

If you already have IsaacLab installed, you only need to ensure `skrl` is available:

```bash
pip install skrl>=1.4.3
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python -c "import skrl; print(f'skrl: {skrl.__version__}')"

# Test SAC modules
python -c "from modules import SAC; print('SAC module imported successfully')"
```

## Integration with IsaacLab

### 1. Ensure IsaacLab is in PYTHONPATH

```bash
# Add to your .bashrc or run before training
export ISAACLAB_PATH=/path/to/IsaacLab
export PYTHONPATH=$ISAACLAB_PATH/source/isaaclab:$PYTHONPATH
export PYTHONPATH=$ISAACLAB_PATH/source/isaaclab_tasks:$PYTHONPATH
export PYTHONPATH=$ISAACLAB_PATH/source/isaaclab_rl:$PYTHONPATH
```

### 2. Run Training

```bash
# From sac_isaaclab directory
python train.py --task Isaac-Ant-Direct-v0 --num_envs 4096 --headless
```

## Troubleshooting

### CUDA Version Mismatch

If you encounter CUDA errors:

```bash
# Check CUDA version
nvcc --version

# Ensure PyTorch CUDA matches system CUDA
python -c "import torch; print(torch.version.cuda)"
```

### NumPy Version Issues

IsaacLab requires `numpy<2`. If you have NumPy 2.x:

```bash
pip install "numpy<2"
```

### skrl Import Errors

If skrl is not found:

```bash
pip install skrl>=1.4.3
```

### IsaacLab Module Not Found

Make sure IsaacLab is properly installed:

```bash
# From IsaacLab directory
./isaaclab.sh --install

# Or manually
pip install -e source/isaaclab
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_rl
```

## GPU Verification

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

## Uninstallation

```bash
pip uninstall sac_isaaclab
```
