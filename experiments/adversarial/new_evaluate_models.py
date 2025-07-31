#!/usr/bin/env python3
"""
Standalone evaluation script for adversarial robustness experiments.
This script can analyze trained models and checkpoints retroactively.
"""

import os
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from .config import get_default_config
from .datasets_adversarial import create_dataloaders
from .utils import setup_logger, get_config_and_results_dir
from .main import (
    run_evaluation_phase,
    create_attack_config,
    create_model_config,
    create_training_config,
    create_sae_config,
    create_dataloaders_from_config
)


def run_evaluation_only(
    results_dir: Path,
    dataset_type: Optional[str] = None,
    enable_llc: bool = True,
    enable_sae: bool = True,
    enable_inference_llc: bool = False,
    enable_sae_checkpoints: bool = False,
) -> Path:
    """Run only the evaluation phase on existing trained models.
    
    This function reuses the existing evaluation logic from main.py
    to ensure consistency and avoid code duplication.
    
    Args:
        results_dir: Path to results directory with trained models
        dataset_type: Optional dataset type for configuration
        enable_llc: Whether to enable LLC analysis
        enable_sae: Whether to enable SAE analysis
        enable_inference_llc: Whether to enable inference-time LLC analysis
        enable_sae_checkpoints: Whether to enable SAE checkpoint analysis
        
    Returns:
        Path to results directory
    """
    # Use the existing evaluation function from main.py
    return run_evaluation_phase(
        results_dir=results_dir,
        dataset_type=dataset_type,
        enable_llc=enable_llc,
        enable_sae=enable_sae,
        enable_inference_llc=enable_inference_llc,
        enable_sae_checkpoints=enable_sae_checkpoints,
    )


if __name__ == "__main__":
    """
    Example usage - Evaluate existing results
    """
    # Example: Evaluate the results from your latest run
    # Use absolute path since the script changes working directory
    project_root = Path(__file__).parent.parent.parent  # Go up to project root
    results_path = project_root / "results/adversarial_robustness/mnist/2025-07-30_17-29_simplecnn_2-class_pgd_quick_test_mnist_simplecnn_with_llc_sae_inference_llc"
    
    if not results_path.exists():
        print(f"Error: Results directory {results_path} not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for: {results_path}")
    else:
        print(f"Evaluating existing results in: {results_path}")
        
        run_evaluation_only(
            results_dir=results_path,
            dataset_type="mnist",
            enable_llc=True,
            enable_sae=True,
            enable_inference_llc=True,
            enable_sae_checkpoints=True,
        ) 

