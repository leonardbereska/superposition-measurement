"""Configuration for adversarial robustness experiments."""

import os
import torch

# Model configuration
MODEL_CONFIG = {
    "model_type": "mlp",
    "hidden_dim": 32,
    "image_size": 28,
}

# Dataset configuration
DATASET_CONFIG = {
    "target_size": 28,
    "binary_digits": (0, 1),  # Use None for multi-class
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "n_epochs": 200,
}

# Adversarial training configuration
ADVERSARIAL_CONFIG = {
    "train_epsilons": [0.0, 0.05, 0.1, 0.15, 0.2],
    "test_epsilons": [0.0, 0.02, 0.05, 0.1, 0.15, 0.2],
    # "train_epsilons": [0.0],
    # "test_epsilons": [0.0],
    "n_runs": 5,  # For statistical significance
}

# SAE configuration
SAE_CONFIG = {
    "expansion_factor": 2,
    "n_epochs": 200,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "l1_lambda": 0.1,
}

# Paths configuration
RESULTS_DIR = "../../results/adversarial_robustness"
DATA_DIR = "../../data"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                    "mps" if torch.backends.mps.is_available() else 
                    "cpu")
# DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
print(f"Data directory: {os.path.abspath(DATA_DIR)}")
print(f"Results directory: {os.path.abspath(RESULTS_DIR)}")
print("\n")
