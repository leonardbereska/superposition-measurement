"""Configuration for adversarial robustness experiments."""

import os
import torch

# Set to True for quick testing, False for full experiments
TESTING_MODE = False
# TESTING_MODE = True

# Model configuration
MODEL_CONFIG = {
    # "model_type": "mlp",
    "model_type": "cnn",
    "hidden_dim": 32,
    "image_size": 28,
}

# Dataset configuration
DATASET_CONFIG = {
    "target_size": 28,
    # "selected_classes": (0, 1),
    # "selected_classes": (0, 1, 2),
    "selected_classes": (0, 1, 2, 3, 4),
    # "selected_classes": None,  # Use all classes
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "n_epochs": 200 if not TESTING_MODE else 5,
}

# Adversarial training configuration
ADVERSARIAL_CONFIG = {
    "train_epsilons": [0.0, 0.05, 0.1, 0.15, 0.2] if not TESTING_MODE else [0.0],
    "test_epsilons": [0.0, 0.02, 0.05, 0.1, 0.15, 0.2] if not TESTING_MODE else [0.0],
    "n_runs": 5 if not TESTING_MODE else 1,  # For statistical significance
}

# SAE configuration
SAE_CONFIG = {
    "expansion_factor": 2,
    "n_epochs": 200 if not TESTING_MODE else 5,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "l1_lambda": 0.1,
}

# Paths configuration
RESULTS_DIR = "../../results/adversarial_robustness"
DATA_DIR = "../../data"
# make results and data directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                    "mps" if torch.backends.mps.is_available() else 
                    "cpu")
# Uncomment to force CPU usage
# DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
print(f"Data directory: {os.path.abspath(DATA_DIR)}")
print(f"Results directory: {os.path.abspath(RESULTS_DIR)}")
print(f"Mode: {'TESTING' if TESTING_MODE else 'FULL EXPERIMENT'}")
print("\n")
