#!/usr/bin/env python3
"""
Standalone training script for adversarial robustness experiments.
This script only handles model training and checkpoint saving.
Evaluation and analysis are handled by separate scripts.
"""

import os
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from .config import get_default_config, update_config
from .datasets_adversarial import create_dataloaders
from .utils import setup_results_dir, save_config, setup_logger, create_directory
from .training import train_model, evaluate_model_performance, save_model
from .attacks import AttackConfig
from .models import ModelConfig, create_model


def run_training_only(
    config: Dict[str, Any], 
    results_dir: Optional[Path] = None
) -> Path:
    """Run only the training phase with checkpoint saving.
    
    Args:
        config: Configuration dictionary
        results_dir: Optional results directory path
        
    Returns:
        Path to results directory
    """
    # Create results directory if not provided
    if results_dir is None:
        results_dir = setup_results_dir(config)
    
    # Set up logger
    logger = setup_logger(results_dir)
    log = logger.info 
    
    log("\n=== Starting Training Phase ===")
    log(f"Results will be saved to: {results_dir}")

    # Save configuration
    save_config(config, results_dir)

    # Log configuration summary
    log("\n=== Configuration Summary ===")
    log(f"\tDataset: {config['dataset']['dataset_type']}")
    log(f"\tNumber of classes: {len(config['dataset']['selected_classes'])}")
    log(f"\tModel type: {config['model']['model_type'].upper()}")
    log(f"\tModel structure: {config['model']}")
    log(f"\tAdversarial attack: {config['adversarial']['attack_type'].upper()}")
    log(f"\tTraining epsilons: {config['adversarial']['train_epsilons']}")
    log(f"\tTest epsilons: {config['adversarial']['test_epsilons']}")
    log(f"\tNumber of runs: {config['adversarial']['n_runs']}")
    
    # Log LLC configuration
    if config['llc'].get('measure_frequency', 0) > 0:
        log(f"\tLLC checkpointing enabled: every {config['llc']['measure_frequency']} epochs")
        log(f"\tLLC hyperparameter tuning: {config['llc'].get('tune_hyperparams', False)}")
    
    # Log hardware information
    log("\n=== Hardware Information ===")
    device = config['device']
    log(f"\tDevice: {device}")
    if device.type == 'cuda':
        log(f"\tGPU: {torch.cuda.get_device_name(device)}")
        log(f"\tCUDA Version: {torch.version.cuda}")
        log(f"\tAvailable GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    log(f"\tPyTorch Version: {torch.__version__}")
    
    # Create dataloaders
    log("\n=== Dataset Information ===")
    train_loader, val_loader = create_dataloaders(config)
    log(f"\tTrain loader size: {len(train_loader.dataset)}")
    log(f"\tValidation loader size: {len(val_loader.dataset)}")
    
    # Create configurations
    attack_config = AttackConfig(
        attack_type=config['adversarial']['attack_type'],
        epsilon=config['adversarial']['test_epsilons'][-1],  # Use max epsilon for attack config
        steps=config['adversarial']['pgd_steps'],
        step_size=config['adversarial']['pgd_step_size']
    )
    
    model_config = ModelConfig(
        model_type=config['model']['model_type'],
        input_channels=config['model']['input_channels'],
        hidden_dim=config['model']['hidden_dim'],
        image_size=config['model']['image_size'],
        output_dim=config['model']['output_dim']
    )
    
    # Train models with different adversarial strengths
    log("\n=== Starting Model Training ===")

    # Log model structure
    model = create_model(
        model_type=model_config.model_type,
        input_channels=model_config.input_channels,
        hidden_dim=model_config.hidden_dim,
        image_size=model_config.image_size,
        output_dim=model_config.output_dim,
        use_nnsight=False
    ).to(device)
    log(f"\tTraining {model_config.model_type.upper()} structure: {model}")
    log(f"\tTraining {model_config.model_type.upper()} parameters: {sum(p.numel() for p in model.parameters())}")

    for epsilon in config['adversarial']['train_epsilons']:
        eps_dir = create_directory(results_dir, f"eps_{epsilon}")
        
        # Run multiple training runs
        for run_idx in range(config['adversarial']['n_runs']):
            run_seed = 42 + run_idx  # Set random seed for reproducibility
            torch.manual_seed(run_seed)
            run_dir = create_directory(eps_dir, f"run_{run_idx}")
            
            log(f"\n--- Training with epsilon={epsilon}, run {run_idx+1}/{config['adversarial']['n_runs']} (seed={run_seed}) ---")
            
            # Train model
            model = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model_config=model_config,
                training_config=config['training'],
                epsilon=epsilon,
                attack_config=attack_config,
                device=config['device'],
                logger=logger,
                save_dir=run_dir  # Pass save directory for checkpoints
            )

            # Save final model
            model_path = run_dir / "model.pt"
            save_model(model, model_config, model_path)
            
            # Evaluate on test epsilons
            robustness_results = evaluate_model_performance(
                model=model,
                dataloader=val_loader,
                epsilons=config['adversarial']['test_epsilons'],
                attack_config=attack_config,
                device=config['device'],
            )
            
            # Log robustness results summary
            log("\tRobustness results:")
            for eps, acc in robustness_results.items():
                log(f"\t\tTest epsilon={eps}: accuracy={acc:.2f}%")
            
            # Save results
            results_path = run_dir / "robustness.json"
            with open(results_path, 'w') as f:
                json.dump(robustness_results, f, indent=4)
    
    log(f"\n=== Training Phase Complete ===")
    log(f"All models trained and saved to: {results_dir}")
    log("Run evaluation script to analyze the trained models.")
    
    return results_dir


def quick_train(
    model_type: str = None, 
    dataset_type: str = "cifar10", 
    testing_mode: bool = True
):
    """Run a quick training experiment with minimal settings.
    
    Args:
        model_type: Optional model type override
        dataset_type: Dataset type ("cifar10" or "mnist")
        testing_mode: Whether to use minimal settings for testing
    """
    # Quick test configuration
    config = get_default_config(dataset_type=dataset_type, testing_mode=testing_mode, selected_classes=(0, 1))
    
    # Customize config for quick test if model_type provided
    if model_type is not None:
        config = update_config(config, {
            'model': {'model_type': model_type}
        })
    
    # Add comment
    config = update_config(config, {
        'comment': f'quick_train_{dataset_type}_{config["model"]["model_type"]}'
    })

    # Print key configuration
    print(f"Training: {config['model']['model_type']}")
    print(f"Dataset: {config['dataset']['dataset_type']}")
    print(f"Attack: {config['adversarial']['attack_type']} with {config['adversarial']['pgd_steps']} steps")
    print(f"Epsilon: {config['adversarial']['train_epsilons']}")
    print(f"Epochs: {config['training']['n_epochs']}")
    print(f"LLC checkpointing: every {config['llc']['measure_frequency']} epochs")

    # Run training phase only
    results_dir = run_training_only(config)
    
    print(f"Quick training completed. Results in: {results_dir}")
    return results_dir


if __name__ == "__main__":
    """
    Example usage - Quick training only
    """
    quick_train(
        model_type='resnet18', 
        dataset_type="mnist", 
        testing_mode=True
    ) 