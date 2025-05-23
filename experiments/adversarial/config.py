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
    if dataset_type.lower() == "cifar10":
        model_config = {
            "model_type": "resnet18",  # Standard for CIFAR-10
            "hidden_dim": None, # using standard resnet18
            "image_size": dataset_config["image_size"],
            "input_channels": dataset_config["input_channels"],
            "output_dim": get_output_dim(dataset_config["selected_classes"]),
        }
    else:  # MNIST
        model_config = {
            "model_type": "cnn",       # Simple CNN for MNIST
            "hidden_dim": 16,          # Smaller network for MNIST
            "image_size": dataset_config["image_size"],
            "input_channels": dataset_config["input_channels"],
            "output_dim": get_output_dim(dataset_config["selected_classes"]),
        }
    
    # Training configuration with dataset-specific defaults
    if dataset_type.lower() == "cifar10":
        # Literature-standard hyperparameters for CIFAR-10 adversarial training
        training_config = {
            "batch_size": 256,     # Standard for CIFAR-10
            "learning_rate": 0.1,      # Standard for CIFAR-10 with SGD
            "n_epochs": 1 if testing_mode else 200,  # Standard: 200 epochs
            "optimizer_type": "sgd",   # SGD with momentum for CIFAR-10
            "weight_decay": 5e-4,      # Standard weight decay for CIFAR-10
            "momentum": 0.9,           # Standard momentum
            "lr_schedule": "multistep",  # Learning rate decay
            "lr_milestones": [100, 150],  # Standard decay schedule
            "lr_gamma": 0.1,           # Decay factor
            "early_stopping_patience": 50,  # Longer patience for adversarial training
            "min_epochs": 50,          # Minimum training before early stopping
            "use_early_stopping": False,  # Standard practice uses fixed epochs
        }
    else:  # MNIST and other datasets
        training_config = {
            "batch_size": 256,     # Standard for MNIST
            "learning_rate": 0.01,     # Standard for MNIST
            "n_epochs": 1 if testing_mode else 100,  # Standard: 100 epochs for MNIST
            "optimizer_type": "sgd",   # SGD is standard
            "weight_decay": 1e-4,      # Standard weight decay for MNIST
            "momentum": 0.9,           # Standard momentum
            "lr_schedule": "multistep",  # Learning rate decay
            "lr_milestones": [50, 75],  # Adjusted for 100 epochs
            "lr_gamma": 0.1,           # Decay factor
            "early_stopping_patience": 20,
            "min_epochs": 0,
            "use_early_stopping": False,  # Standard practice uses fixed epochs
        }
    
    # Adversarial training configuration with dataset-specific epsilon values
    if dataset_type.lower() == "mnist":
        # Standard epsilon values for MNIST (in [0,1] pixel range)
        train_epsilons = [0.0, 0.1, 0.2, 0.3] if not testing_mode else [0.0, 0.1]
        test_epsilons = [0.0, 0.1, 0.2, 0.3] if not testing_mode else [0.0, 0.1]
        pgd_steps = 40       # Standard: 40 steps for MNIST
        pgd_alpha = 0.01     # Standard step size for MNIST
    elif dataset_type.lower() == "cifar10":
        # Standard epsilon values for CIFAR-10 (in [0,1] pixel range)
        # 8/255 ≈ 0.031 is the de facto standard for CIFAR-10
        train_epsilons = [0.0, 0.01, 0.02, 0.03] if not testing_mode else [0.0, 0.01]
        test_epsilons = [0.0, 0.01, 0.02, 0.03] if not testing_mode else [0.0, 0.01]
        pgd_steps = 10       # Standard: 10 steps for CIFAR-10
        pgd_alpha = 2/255    # Standard step size for CIFAR-10
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    adversarial_config = {
        "train_epsilons": train_epsilons,
        "test_epsilons": test_epsilons,
        "n_runs": 1 if testing_mode else 3,  # Number of runs per epsilon
        "attack_type": "pgd",  # PGD is the standard attack type
        "pgd_steps": pgd_steps,  # Number of steps for PGD attack
        "pgd_alpha": pgd_alpha,  # Step size for PGD
        "pgd_random_start": True  # Standard: random initialization
    }
    
    # SAE configuration
    sae_config = {
        "expansion_factor": 4,  
        "n_epochs": 1 if testing_mode else 800,  # early stopping usually around 500
        "learning_rate": 1e-3,
        "batch_size": 128,
        "l1_lambda": 0.1,
    }

    comment = ""
    
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
        "comment": comment,
    }
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
        
        # Update adversarial config with dataset-specific epsilon values
        # Only if adversarial config wasn't explicitly provided in updates
        if "adversarial" not in updates or not any(k in updates.get("adversarial", {}) 
                                                for k in ["train_epsilons", "test_epsilons"]):
            if dataset_type.lower() == "mnist":
                # Standard epsilon values for MNIST
                updated_config["adversarial"]["train_epsilons"] = [0.0, 0.3]
                updated_config["adversarial"]["test_epsilons"] = [0.0, 0.3]
                updated_config["adversarial"]["pgd_alpha"] = 0.01
                updated_config["adversarial"]["pgd_steps"] = 40  # Standard for MNIST
            else:  # CIFAR-10
                # Standard epsilon values for CIFAR-10
                updated_config["adversarial"]["train_epsilons"] = [0.0, 0.031]
                updated_config["adversarial"]["test_epsilons"] = [0.0, 0.031]
                updated_config["adversarial"]["pgd_alpha"] = 0.008
                updated_config["adversarial"]["pgd_steps"] = 10  # Standard for CIFAR-10
            
            # Apply testing mode adjustments if needed
            if updated_config["testing_mode"]:
                updated_config["adversarial"]["train_epsilons"] = updated_config["adversarial"]["train_epsilons"][:2]
                updated_config["adversarial"]["test_epsilons"] = updated_config["adversarial"]["test_epsilons"][:2]
    
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