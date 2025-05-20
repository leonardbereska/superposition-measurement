"""Utility functions for adversarial robustness experiments."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Tuple


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
    
    # Create directory name with timestamp, model type, and class count
    dir_name = f"{timestamp}_{model_type}_{n_classes}-class"
    
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

def load_config(results_dir: Path) -> Dict:
    """Load experiment configuration.
    
    Args:
        results_dir: Directory containing experiment configuration
        
    Returns:
        Dictionary with configuration
    """
    metadata_path = results_dir / "config.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        return {}



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