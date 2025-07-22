#!/usr/bin/env python3
"""
Simple script to analyze existing results and generate plots.
This script can be run on any results directory that has evaluation data.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from .main import run_analysis_phase


def analyze_existing_results(
    results_dir: str,
    dataset_type: Optional[str] = None,
    plot_args: Optional[Dict[str, Any]] = None
):
    """Analyze existing results and generate plots.
    
    Args:
        results_dir: Path to results directory
        dataset_type: Optional dataset type for configuration
        plot_args: Optional plot arguments
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory {results_path} not found!")
        return
    
    print(f"Analyzing existing results in: {results_path}")
    
    # Run analysis phase
    run_analysis_phase(
        results_dir=results_path,
        dataset_type=dataset_type,
        plot_args=plot_args
    )
    
    print(f"Analysis complete! Check the plots directory in: {results_path}")


if __name__ == "__main__":
    """
    Example usage - Analyze existing results
    """
    # Example: Analyze the results from your latest run
    analyze_existing_results(
        results_dir="./results/adversarial_robustness/mnist/2025-07-22_12-49_resnet18_2-class_pgd_quick_test_mnist_resnet18_with_llc_sae_inference_llc",
        dataset_type="mnist"
    ) 