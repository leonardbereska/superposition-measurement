#!/usr/bin/env python3
"""
Memory-optimized evaluation script for adversarial robustness experiments.
This script includes optimizations to prevent OOM errors during SAE and LLC analysis.
"""

import os
import torch
import json
import logging
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from .config import get_default_config
from .datasets_adversarial import create_dataloaders
from .utils import setup_logger, get_config_and_results_dir
from .evaluation import (
    analyze_checkpoints_with_sae, 
    analyze_checkpoints_with_llc, 
    analyze_checkpoints_combined_sae_llc,
    measure_inference_time_llc
)
from .training import load_model
from .attacks import get_adversarial_dataloader
from .sae import SAEConfig


def create_memory_optimized_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a memory-optimized configuration to prevent OOM errors."""
    optimized_config = config.copy()
    
    # Reduce SAE parameters for memory efficiency
    if 'sae' in optimized_config:
        optimized_config['sae'] = optimized_config['sae'].copy()
        optimized_config['sae']['batch_size'] = min(64, optimized_config['sae']['batch_size'])
        optimized_config['sae']['n_epochs'] = min(12, optimized_config['sae']['n_epochs'])  # Reduce epochs for testing
        optimized_config['sae']['expansion_factor'] = min(2.0, optimized_config['sae']['expansion_factor'])  # Reduce expansion
    
    # Reduce LLC parameters for memory efficiency
    if 'llc' in optimized_config:
        optimized_config['llc'] = optimized_config['llc'].copy()
        optimized_config['llc']['num_draws'] = min(200, optimized_config['llc'].get('num_draws', 1500))
        optimized_config['llc']['num_chains'] = min(2, optimized_config['llc'].get('num_chains', 3))
        optimized_config['llc']['epsilon_samples'] = min(3, optimized_config['llc'].get('epsilon_samples', 5))
        optimized_config['llc']['beta_samples'] = min(3, optimized_config['llc'].get('beta_samples', 5))
    
    return optimized_config


def run_memory_optimized_evaluation(
    results_dir: Path,
    dataset_type: Optional[str] = None,
    enable_llc: bool = True,
    enable_sae: bool = True,
    enable_inference_llc: bool = False,
    enable_sae_checkpoints: bool = False,
    enable_llc_checkpoints: bool = True,
    enable_combined_checkpoints: bool = False,
    max_samples_per_analysis: int = 5000,  # Limit samples to prevent OOM
    process_one_epsilon_at_a_time: bool = True  # Process epsilons sequentially
) -> Path:
    """Run memory-optimized evaluation on existing trained models.
    
    Args:
        results_dir: Path to results directory with trained models
        dataset_type: Optional dataset type for configuration
        enable_llc: Whether to enable LLC analysis
        enable_sae: Whether to enable SAE analysis
        enable_inference_llc: Whether to enable inference-time LLC analysis
        enable_sae_checkpoints: Whether to enable SAE checkpoint analysis
        enable_llc_checkpoints: Whether to enable LLC checkpoint analysis
        enable_combined_checkpoints: Whether to enable combined SAE+LLC checkpoint analysis
        max_samples_per_analysis: Maximum samples to use per analysis to prevent OOM
        process_one_epsilon_at_a_time: Whether to process epsilons sequentially to save memory
        
    Returns:
        Path to results directory
    """
    # Load and optimize configuration
    config = get_default_config(dataset_type=dataset_type)
    config = create_memory_optimized_config(config)
    config, results_dir = get_config_and_results_dir(
        base_dir=Path(config['base_dir']), 
        results_dir=results_dir
    )
    
    logger = setup_logger(results_dir)
    log = logger.info

    log("\n=== Starting Memory-Optimized Evaluation Phase ===")
    log(f"Results directory: {results_dir}")
    log(f"Max samples per analysis: {max_samples_per_analysis}")
    log(f"Process epsilons sequentially: {process_one_epsilon_at_a_time}")
    log(f"SAE analysis: {'enabled' if enable_sae else 'disabled'}")
    log(f"SAE checkpoint analysis: {'enabled' if enable_sae_checkpoints else 'disabled'}")
    log(f"LLC analysis: {'enabled' if enable_llc else 'disabled'}")
    log(f"LLC checkpoint analysis: {'enabled' if enable_llc_checkpoints else 'disabled'}")
    log(f"Combined checkpoint analysis: {'enabled' if enable_combined_checkpoints else 'disabled'}")
    log(f"Inference-time LLC analysis: {'enabled' if enable_inference_llc else 'disabled'}")

    # Create dataloaders with reduced batch size for memory efficiency
    train_loader, _ = create_dataloaders(config)
    
    # Create SAE config with memory optimizations
    sae_config = None
    if enable_sae or enable_sae_checkpoints:
        sae_config = SAEConfig(
            expansion_factor=config['sae']['expansion_factor'],
            l1_lambda=config['sae']['l1_lambda'],
            batch_size=config['sae']['batch_size'],
            learning_rate=config['sae']['learning_rate'],
            n_epochs=config['sae']['n_epochs']
        )

    # Get layer names for SAE analysis
    layer_names = []
    if enable_sae or enable_sae_checkpoints:
        from .models import get_layer_names_for_model
        layer_names = get_layer_names_for_model(config['model']['model_type'])

    # Process epsilons
    epsilons_to_process = config['adversarial']['train_epsilons']
    if process_one_epsilon_at_a_time:
        log("Processing epsilons sequentially to save memory...")
    
    for epsilon in epsilons_to_process:
        eps_dir = results_dir / f"eps_{epsilon}"
        if not eps_dir.exists():
            log(f"Warning: Epsilon directory {eps_dir} not found, skipping...")
            continue
            
        log(f"\n--- Evaluating epsilon={epsilon} ---")
        
        # Process each run
        for run_idx in range(config['adversarial']['n_runs']):
            run_dir = eps_dir / f"run_{run_idx}"
            if not run_dir.exists():
                log(f"Warning: Run directory {run_dir} not found, skipping...")
                continue
                
            log(f"\n--- Evaluating run {run_idx+1}/{config['adversarial']['n_runs']} ---")
            
            # Load trained model
            model_path = run_dir / "model.pt"
            if not model_path.exists():
                log(f"Warning: Model file {model_path} not found, skipping...")
                continue
                
            model, model_config = load_model(model_path=model_path, device=config['device'])
            
            # Clear GPU memory before each analysis
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create adversarial dataloader for inference-time LLC
            adv_loader = None
            if enable_inference_llc:
                from .attacks import AttackConfig
                attack_config = AttackConfig(
                    attack_type=config['adversarial']['attack_type'],
                    epsilon=epsilon,
                    steps=config['adversarial']['pgd_steps'],
                    step_size=config['adversarial']['pgd_step_size']
                )
                adv_loader = get_adversarial_dataloader(model, train_loader, attack_config, epsilon)

            # Checkpoint analysis with memory management
            checkpoint_dir = run_dir / "checkpoints"
            if checkpoint_dir.exists():
                log(f"Found checkpoints in {checkpoint_dir}")
                
                # Combined SAE + LLC checkpoint analysis
                if enable_combined_checkpoints:
                    log("Performing combined SAE + LLC checkpoint analysis...")
                    try:
                        combined_results = analyze_checkpoints_combined_sae_llc(
                            checkpoint_dir=checkpoint_dir,
                            train_loader=train_loader,
                            sae_config=sae_config,
                            llc_config=config['llc'],
                            layer_names=layer_names,
                            epsilons_to_test=[0.0, epsilon],
                            device=config['device'],
                            logger=logger
                        )
                        
                        # Save combined results
                        combined_path = run_dir / "combined_sae_llc_analysis_results.json"
                        with open(combined_path, 'w') as f:
                            json.dump(combined_results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
                        log(f"Combined analysis results saved to: {combined_path}")
                        
                        # Clear memory after combined analysis
                        del combined_results
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                    except Exception as e:
                        log(f"Error in combined checkpoint analysis: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                
                # Individual checkpoint analyses
                else:
                    # LLC checkpoint analysis
                    if enable_llc_checkpoints:
                        log("Performing LLC checkpoint analysis...")
                        try:
                            llc_results = analyze_checkpoints_with_llc(
                                checkpoint_dir=checkpoint_dir,
                                train_loader=train_loader,
                                llc_config=config['llc'],
                                epsilons_to_test=[0.0, epsilon],
                                device=config['device'],
                                logger=logger
                            )
                            log(f"LLC checkpoint analysis completed for {len(llc_results.get('epsilon_analysis', {}))} epsilons")
                            
                            # Clear memory after LLC analysis
                            del llc_results
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            
                        except Exception as e:
                            log(f"Error in LLC checkpoint analysis: {e}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                    
                    # SAE checkpoint analysis
                    if enable_sae_checkpoints:
                        log("Performing SAE checkpoint analysis...")
                        try:
                            sae_results = analyze_checkpoints_with_sae(
                                checkpoint_dir=checkpoint_dir,
                                train_loader=train_loader,
                                sae_config=sae_config,
                                layer_names=layer_names,
                                epsilons_to_test=[0.0, epsilon],
                                device=config['device'],
                                logger=logger
                            )
                            log(f"SAE checkpoint analysis completed for {len(sae_results.get('epsilon_analysis', {}))} epsilons")
                            
                            # Clear memory after SAE analysis
                            del sae_results
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            
                        except Exception as e:
                            log(f"Error in SAE checkpoint analysis: {e}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
            
            # Inference-time LLC analysis
            if enable_inference_llc and adv_loader is not None:
                log("Performing inference-time LLC analysis...")
                try:
                    dataloaders = {
                        'clean': train_loader,
                        f'eps_{epsilon}': adv_loader
                    }
                    
                    inference_results = measure_inference_time_llc(
                        model=model,
                        dataloaders=dataloaders,
                        llc_config=config['llc'],
                        save_dir=run_dir / "inference_llc",
                        logger=logger
                    )
                    
                    if inference_results:
                        # Save inference results
                        inference_path = run_dir / "inference_llc_results.json"
                        with open(inference_path, 'w') as f:
                            json.dump(inference_results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
                        log(f"Inference-time LLC results saved to: {inference_path}")
                        
                        # Clear memory after inference analysis
                        del inference_results
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    log(f"Error in inference-time LLC analysis: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Clear model from memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            log(f"Completed evaluation for epsilon={epsilon}, run={run_idx}")
        
        # Clear memory between epsilons if processing sequentially
        if process_one_epsilon_at_a_time:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            log(f"Cleared memory after processing epsilon={epsilon}")

    log(f"\n=== Memory-Optimized Evaluation Phase Complete ===")
    log(f"All evaluations completed for: {results_dir}")
    log("Run analysis script to generate plots and summaries.")
    
    return results_dir


def evaluate_existing_results_memory_optimized(
    results_dir: str,
    dataset_type: Optional[str] = None,
    enable_llc: bool = True,
    enable_sae: bool = True,
    enable_inference_llc: bool = False,
    enable_sae_checkpoints: bool = False,
    enable_llc_checkpoints: bool = True,
    enable_combined_checkpoints: bool = False,
    max_samples_per_analysis: int = 5000,
    process_one_epsilon_at_a_time: bool = True
):
    """Evaluate existing results directory with memory optimizations.
    
    Args:
        results_dir: Path to results directory
        dataset_type: Optional dataset type
        enable_llc: Whether to enable LLC analysis
        enable_sae: Whether to enable SAE analysis
        enable_inference_llc: Whether to enable inference-time LLC analysis
        enable_sae_checkpoints: Whether to enable SAE checkpoint analysis
        enable_llc_checkpoints: Whether to enable LLC checkpoint analysis
        enable_combined_checkpoints: Whether to enable combined SAE+LLC checkpoint analysis
        max_samples_per_analysis: Maximum samples to use per analysis
        process_one_epsilon_at_a_time: Whether to process epsilons sequentially
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory {results_path} not found!")
        return
    
    print(f"Evaluating existing results in: {results_path}")
    print(f"Memory optimizations: max_samples={max_samples_per_analysis}, sequential_epsilons={process_one_epsilon_at_a_time}")
    print(f"LLC analysis: {'enabled' if enable_llc else 'disabled'}")
    print(f"SAE analysis: {'enabled' if enable_sae else 'disabled'}")
    print(f"Checkpoint analysis: {'enabled' if (enable_sae_checkpoints or enable_llc_checkpoints or enable_combined_checkpoints) else 'disabled'}")
    print(f"Inference-time LLC: {'enabled' if enable_inference_llc else 'disabled'}")
    
    run_memory_optimized_evaluation(
        results_dir=results_path,
        dataset_type=dataset_type,
        enable_llc=enable_llc,
        enable_sae=enable_sae,
        enable_inference_llc=enable_inference_llc,
        enable_sae_checkpoints=enable_sae_checkpoints,
        enable_llc_checkpoints=enable_llc_checkpoints,
        enable_combined_checkpoints=enable_combined_checkpoints,
        max_samples_per_analysis=max_samples_per_analysis,
        process_one_epsilon_at_a_time=process_one_epsilon_at_a_time
    )


if __name__ == "__main__":
    """
    Example usage - Memory-optimized evaluation of existing results
    """
    # Example: Evaluate the results from your latest run with memory optimizations
    evaluate_existing_results_memory_optimized(
        results_dir="./results/adversarial_robustness/mnist/2025-07-22_12-49_resnet18_2-class_pgd_quick_test_mnist_resnet18_with_llc_sae_inference_llc",
        dataset_type="mnist",
        enable_llc=True,
        enable_sae=True,
        enable_inference_llc=True,
        enable_sae_checkpoints=True,
        enable_llc_checkpoints=True,
        enable_combined_checkpoints=False,
        max_samples_per_analysis=3000,  # Reduce samples to prevent OOM
        process_one_epsilon_at_a_time=True  # Process epsilons sequentially
    ) 