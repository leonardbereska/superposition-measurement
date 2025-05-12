#!/bin/bash
# setup_conda_env.sh

# Create a new conda environment
conda create -n superposition python=3.10 -y

# Activate the environment
conda activate superposition

# Install PyTorch with CUDA support
# Modify cuda version as needed (11.8, 12.1, etc.)
# conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pytorch torchvision -y  # for cpu

# Install core data science packages
conda install numpy matplotlib tqdm ipykernel jupyter -y

# Install Hugging Face datasets
pip install datasets

# Install nnsight
pip install nnsight

# Install dictionary learning
pip install dictionary-learning

