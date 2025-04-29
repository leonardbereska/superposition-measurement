# %%
"""Configuration utilities for adversarial robustness experiments."""

import os
import torch
from typing import Dict, Any, List, Optional

# %%
def get_default_device() -> torch.device:
    """Get default device for computations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# %%
def get_default_config(testing_mode: bool = False) -> Dict[str, Any]:
    """Get default configuration for experiments.
    
    Args:
        testing_mode: Whether to use minimal settings for testing
        
    Returns:
        Configuration dictionary with all parameters
    """
    # Base directories
    results_dir = "../../results/adversarial_robustness"
    data_dir = "../../data"
    
    # Create directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Device configuration
    device = get_default_device()
    
    # Model configuration
    model_config = {
        "model_type": "mlp",  # Options: "mlp", "cnn"
        "hidden_dim": 32,
        "image_size": 28,
    }
    
    # Dataset configuration
    dataset_config = {
        "dataset_type": "mnist",
        "target_size": 28,
        "selected_classes": (0, 1),  # Binary classification
        # "selected_classes": None,  # Use all classes
    }
    
    # Training configuration
    training_config = {
        "batch_size": 64,
        "learning_rate": 1e-3,
        # Use shorter epochs in testing mode
        "n_epochs": 5 if testing_mode else 100,
    }
    
    # Adversarial training configuration
    adversarial_config = {
        "train_epsilons": [0.0, 0.05, 0.1, 0.15, 0.2] if not testing_mode else [0.0, 0.1],
        "test_epsilons": [0.0, 0.02, 0.05, 0.1, 0.15, 0.2] if not testing_mode else [0.0, 0.1],
        "n_runs": 1 if testing_mode else 5,  # Number of runs per epsilon
    }
    
    # SAE configuration
    sae_config = {
        "expansion_factor": 2,
        "n_epochs": 5 if testing_mode else 100,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "l1_lambda": 0.1,
    }
    
    # Assemble complete configuration
    config = {
        "device": device,
        "results_dir": results_dir,
        "data_dir": data_dir,
        "model": model_config,
        "dataset": dataset_config,
        "training": training_config,
        "adversarial": adversarial_config,
        "sae": sae_config,
        "testing_mode": testing_mode,
    }
    
    # Print basic information
    print(f"Using device: {device}")
    print(f"Data directory: {os.path.abspath(data_dir)}")
    print(f"Results directory: {os.path.abspath(results_dir)}")
    print(f"Mode: {'TESTING' if testing_mode else 'FULL EXPERIMENT'}")
    
    return config

# %%
def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values.
    
    Args:
        config: Base configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    # Create a copy of the config
    updated_config = config.copy()
    
    # Update top-level keys
    for key, value in updates.items():
        if key in updated_config:
            if isinstance(value, dict) and isinstance(updated_config[key], dict):
                # Recursively update nested dictionaries
                updated_config[key] = update_config(updated_config[key], value)
            else:
                # Replace value
                updated_config[key] = value
    
    return updated_config

# %%
# Example usage
if __name__ == "__main__":
    # Get default config
    config = get_default_config(testing_mode=True)
    
    # Update config with custom values
    custom_updates = {
        "model": {"hidden_dim": 64},
        "dataset": {"selected_classes": (0, 1, 2)},
    }
    
    updated_config = update_config(config, custom_updates)
    print("\nUpdated configuration:")
    print(f"Hidden dimension: {updated_config['model']['hidden_dim']}")
    print(f"Selected classes: {updated_config['dataset']['selected_classes']}")