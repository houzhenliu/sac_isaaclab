from setuptools import setup, find_packages

setup(
    name="sac_isaaclab",
    version="0.1.0",
    description="Custom SAC implementation for IsaacLab",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies matching IsaacLab requirements
        "numpy<2",                    # User has 1.26.0
        "torch>=2.7",                 # User has 2.7.0+cu128
        "gymnasium==1.2.1",           # User has 1.2.1
        
        # RL framework
        "skrl>=1.4.3",                # User has 1.4.3
        
        # Utilities
        "tensorboard>=2.13.0",
        "tqdm",
        "rich",
        
        # IsaacLab dependencies (will be installed if not present)
        "hydra-core",
        "omegaconf",
        "prettytable==3.3.0",
        "toml",
        "packaging<24",
        "protobuf>=4.25.8,!=5.26.0",
    ],
    extras_require={
        "isaaclab": [
            "isaaclab",
            "isaaclab_tasks",
            "isaaclab_rl",
        ],
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cu128",
    ],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
    ],
)
