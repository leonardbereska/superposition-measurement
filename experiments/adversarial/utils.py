"""Utility functions for adversarial robustness experiments."""

import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, Union

def create_directory(base_dir: Path, dir_name: str) -> Path:
    """Create a directory with the given name under the base directory.
    
    Args:
        base_dir: Base directory path
        dir_name: Name of the directory to create
        
    Returns:
        Path to the created directory
    """
    new_dir = base_dir / dir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def setup_logger(results_dir: Path, log_filename: str = "experiment.log") -> logging.Logger:
    """Set up a logger that writes to both console and file.
    
    Args:
        results_dir: Directory to save log file
        log_filename: Name of log file
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("adversarial_experiment")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # Create file handler
    log_path = results_dir / log_filename
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_results_dir(config: Dict[str, Any]) -> Path:
    """Create and return results directory based on configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Path to results directory
    """
    # Get base directory from config
    base_dir = Path(config['base_dir'])
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Get model type and class count for directory name
    model_type = config['model']['model_type']
    n_classes = len(config['dataset']['selected_classes'])
    attack_type = config['adversarial']['attack_type']
    comment = config['comment']
    # Create directory name with timestamp, model type, and class count
    if comment is not None and comment != "":
        dir_name = f"{timestamp}_{model_type}_{n_classes}-class_{attack_type}_{comment}"
    else:
        dir_name = f"{timestamp}_{model_type}_{n_classes}-class_{attack_type}"
    
    # Create full path including dataset name
    results_dir = base_dir / dir_name
    
    # Create directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir

def get_config_and_results_dir(base_dir: Path, results_dir: Optional[Path] = None, search_string: Optional[Path] = None) -> Tuple[Dict[str, Any], Path]:
    """Get config and results directory.
    
    Args:
        results_dir: Optional results directory
        search_string: Optional search string to find results directory

    Returns:
        Tuple of config and results directory
    """
    if results_dir is None:
        results_dir = find_results_dir(base_dir=base_dir, search_string=search_string)
    config = load_config(results_dir)
    print(f"Using results directory: {results_dir}")
    return config, results_dir

def find_latest_results_dir(base_dir: Path) -> Optional[Path]:
    """Find most recent results directory.
    
    Returns:
        Path to latest results directory or None if not found
    """
    results_dirs = sorted(base_dir.glob("*"), key=os.path.getmtime)
    # print base_dir
    print(f"Base directory: {base_dir}")
    print(f"Found {len(results_dirs)} results directories.")
    if not results_dirs:
        print("No results directories found.")
        return None
    
    return results_dirs[-1]

def find_results_dir(base_dir: Path, search_string: Optional[Path] = None) -> Path:
    """Find results directory.
    
    Args:
        search_string: Optional search string to find results directory

    Returns:
        Path to results directory
    """
    # Determine results directory if not provided
    if search_string is None:
        results_dir = find_latest_results_dir(base_dir)
        print(f"Results directory: {results_dir}")
        if results_dir is None:
            raise ValueError("No results directory found. Please run training phase first.")
    else: 
        # the given results_dir is for example "mlp_2-class"
        # we need to find a results directory that contains this
        # so search for a directory that contains this string
        results_dir = next((d for d in base_dir.glob("*") if search_string in d.name), None)
        if results_dir is None:
            raise ValueError(f"No results directory found containing {search_string}")
    return results_dir

def save_config(config: Dict[str, Any], results_dir: Path) -> None:
    """Save configuration to file.
    
    Args:
        config: Experiment configuration
        results_dir: Directory to save configuration
    """
    # Create a JSON-serializable copy of the config
    json_config = {
        k: (str(v) if isinstance(v, torch.device) else v)
        for k, v in config.items()
    }
    
    # Save as JSON
    with open(results_dir / "config.json", 'w') as f:
        json.dump(json_config, f, indent=4, default=json_serializer)

def load_config(results_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Configuration dictionary
    """
    # Convert to Path object if string
    results_dir = Path(results_dir)
    metadata_path = results_dir / "config.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Config file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)



def json_serializer(obj):
    """Serialize objects for JSON serialization."""
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int64) or isinstance(obj, np.int32):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Add more type conversions as needed
    return str(obj)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

