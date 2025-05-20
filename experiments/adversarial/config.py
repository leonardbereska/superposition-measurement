# %%
"""Configuration utilities for adversarial robustness experiments."""

import os
import torch
from typing import Dict, Any, List, Optional, Tuple

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
def get_dataset_config(dataset_type: str, selected_classes: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
    """Get dataset-specific configuration.
    
    Args:
        dataset_type: Type of dataset ('mnist' or 'cifar10')
        selected_classes: Classes to use (None for all classes)
        
    Returns:
        Dataset configuration dictionary
    """
    if dataset_type.lower() == "mnist":
        return {
            "dataset_type": "mnist",
            "image_size": 28,
            "input_channels": 1,   # Grayscale
            "selected_classes": selected_classes,
        }
    elif dataset_type.lower() == "cifar10":
        return {
            "dataset_type": "cifar10",
            "image_size": 32,
            "input_channels": 3,   # RGB
            "selected_classes": selected_classes,
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
def get_output_dim(selected_classes):
    """Get output dimension based on selected classes."""
    num_classes = len(selected_classes)
    return num_classes if num_classes > 2 else 1

# %%
def get_default_config(testing_mode: bool = False, dataset_type: str = "mnist", selected_classes: Tuple[int, ...] = (0, 1)) -> Dict[str, Any]:
    """Get default configuration for experiments.
    
    Args:
        testing_mode: Whether to use minimal settings for testing
        dataset_type: Type of dataset to use ('mnist' or 'cifar10')
        
    Returns:
        Configuration dictionary with all parameters
    """
    
    
    # Device configuration
    device = get_default_device()
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_type, selected_classes=selected_classes)
    
    # Base directories
    base_dir = f"../../results/adversarial_robustness/{dataset_type.lower()}"
    data_dir = "../../data"
    
    # Create directories if they don't exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Model configuration - use dataset parameters
    model_config = {
        "model_type": "mlp",  # Options: "mlp", "cnn"
        "hidden_dim": 32,
        "image_size": dataset_config["image_size"],
        "input_channels": dataset_config["input_channels"],
        "output_dim": get_output_dim(dataset_config["selected_classes"]),
    }
    
    # Training configuration
    training_config = {
        "batch_size": 128,
        "learning_rate": 1e-3,
        # Use shorter epochs in testing mode
        "n_epochs": 2 if testing_mode else 300,  # early stopping seemed to make results less reliable
    }
    
    # Adversarial training configuration
    adversarial_config = {
        "train_epsilons": [0.0, 0.05, 0.1, 0.15, 0.2] if not testing_mode else [0.0, 0.1],
        "test_epsilons": [0.0, 0.02, 0.05, 0.1, 0.15, 0.2] if not testing_mode else [0.0, 0.1],
        "n_runs": 1 if testing_mode else 5,  # Number of runs per epsilon
        "attack_type": "pgd",  # Default attack type, fgsm is also available
        "pgd_steps": 10,        # Number of steps for PGD attack
        "pgd_alpha": None,      # Step size for PGD (None = auto-calculate as epsilon/4)
        "pgd_random_start": True  # Whether to use random initialization for PGD
    }
    
    # SAE configuration
    sae_config = {
        "expansion_factor": 4,  
        "n_epochs": 5 if testing_mode else 800,  # early stopping usually around 500
        "learning_rate": 1e-3,
        "batch_size": 128,
        "l1_lambda": 0.1,
    }
    
    # Assemble complete configuration
    config = {
        "device": device,
        "base_dir": base_dir,
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
    print(f"Dataset: {dataset_type.upper()}")
    print(f"Image size: {dataset_config['image_size']}x{dataset_config['image_size']}, {dataset_config['input_channels']} channels")
    print(f"Data directory: {os.path.abspath(data_dir)}")
    print(f"Base directory: {os.path.abspath(base_dir)}")
    print(f"Mode: {'TESTING' if testing_mode else 'FULL EXPERIMENT'}")
    print(f"Attack type: {adversarial_config['attack_type']}")
    
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
    
    # If dataset type was updated, update related parameters
    if "dataset" in updates and "dataset_type" in updates["dataset"]:
        dataset_type = updates["dataset"]["dataset_type"]
        new_dataset_config = get_dataset_config(
            dataset_type, 
            selected_classes=updated_config["dataset"].get("selected_classes")
        )
        
        # Update dataset config
        updated_config["dataset"].update(new_dataset_config)
        
        # Update model config to match dataset
        updated_config["model"]["image_size"] = new_dataset_config["image_size"]
        updated_config["model"]["input_channels"] = new_dataset_config["input_channels"]
    
    # If selected classes were updated, update output_dim to match
    if "dataset" in updates and "selected_classes" in updates["dataset"]:
        selected_classes = updated_config["dataset"]["selected_classes"]
        # Set output_dim to number of classes for multi-class, or 1 for binary
        updated_config["model"]["output_dim"] = get_output_dim(selected_classes)
    
    return updated_config

# %%
# Example usage
if __name__ == "__main__":
    # Get default config
    config = get_default_config(dataset_type="mnist", testing_mode=True)
    
    # Update config with custom values
    custom_updates = {
        "model": {"hidden_dim": 64},
        "dataset": {"dataset_type": "mnist", "selected_classes": (0, 1, 2)},
    }
    
    updated_config = update_config(config, custom_updates)
    print("\nUpdated configuration:")
    print(f"Dataset: {updated_config['dataset']['dataset_type']}")
    print(f"Image size: {updated_config['model']['image_size']}")
    print(f"Input channels: {updated_config['model']['input_channels']}")
    print(f"Hidden dimension: {updated_config['model']['hidden_dim']}")
    print(f"Selected classes: {updated_config['dataset']['selected_classes']}")