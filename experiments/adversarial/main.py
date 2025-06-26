# %%
"""Main script for running adversarial robustness experiments with SAE and LLC analysis."""
from analysis import ScientificPlotStyle
import matplotlib.pyplot as plt
import os
import torch
import json
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from evaluation import analyze_checkpoints_with_sae, measure_superposition, analyze_checkpoints_with_llc, measure_inference_time_llc, analyze_checkpoints_combined_sae_llc

from analysis import plot_combined_llc_sae_evolution, plot_aggregated_llc_sae_evolution, plot_llc_vs_sae_correlation_over_time, plot_llc_sae_dual_axis_evolution
from training import create_adversarial_dataloader
from training import load_checkpoint
from attacks import AttackConfig
from utils import get_config_and_results_dir, setup_logger
from evaluation import estimate_llc

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import configuration, datasets, utils
from config import get_default_config, update_config
from datasets_adversarial import create_dataloaders
from utils import json_serializer, setup_results_dir, save_config, get_config_and_results_dir, setup_logger, create_directory

# Import components for different phases 
from training import train_model, evaluate_model_performance, load_model, save_model, TrainingConfig
from attacks import AttackConfig, generate_adversarial_examples, get_adversarial_dataloader
from models import ModelConfig, create_model
from sae import SAEConfig

from analysis import (
    generate_plots, 
    plot_training_evolution, 
    plot_sampling_evolution, 
    plot_llc_across_training_multiple_eps,
    plot_llc_traces, 
    plot_llc_distribution, 
    plot_llc_mean_std,
    plot_combined_sae_llc_correlation,
    plot_clean_vs_adversarial_llc,
    plot_combined_llc_sae_evolution,
    plot_llc_vs_sae_correlation_over_time,
    plot_llc_sae_dual_axis_evolution,
    plot_aggregated_llc_sae_evolution
)


def create_attack_config(config: Dict[str, Any]) -> AttackConfig:
    """Create attack configuration from dictionary config."""
    return AttackConfig(
        attack_type=config['adversarial']['attack_type'],
        pgd_steps=config['adversarial']['pgd_steps'],
        pgd_alpha=config['adversarial']['pgd_alpha'],
        pgd_random_start=config['adversarial']['pgd_random_start']
    )

def create_model_config(config: Dict[str, Any]) -> ModelConfig:
    """Create model configuration from dictionary config."""
    return ModelConfig(
        model_type=config['model']['model_type'],
        hidden_dim=config['model']['hidden_dim'],
        input_channels=config['dataset']['input_channels'],
        image_size=config['dataset']['image_size'],
        output_dim=config['model']['output_dim']
    )

def create_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """Create training configuration from dictionary config."""
    return TrainingConfig(
        n_epochs=config['training']['n_epochs'],
        learning_rate=config['training']['learning_rate'],
        optimizer_type=config['training'].get('optimizer_type', 'adam'),
        weight_decay=config['training'].get('weight_decay', 0.0),
        momentum=config['training'].get('momentum', 0.9),
        lr_schedule=config['training'].get('lr_schedule', None),
        lr_milestones=config['training'].get('lr_milestones', None),
        lr_gamma=config['training'].get('lr_gamma', 0.1),
        early_stopping_patience=config['training'].get('early_stopping_patience', 20),
        min_epochs=config['training'].get('min_epochs', 0),
        use_early_stopping=config['training'].get('use_early_stopping', True),
        save_checkpoints=config['llc'].get('measure_frequency', 0) > 0,
        checkpoint_frequency=config['llc'].get('measure_frequency', 5)
    )

def create_sae_config(config: Dict[str, Any]) -> SAEConfig:
    """Create SAE configuration from dictionary config."""
    return SAEConfig(
        expansion_factor=config['sae']['expansion_factor'],
        l1_lambda=config['sae']['l1_lambda'],
        batch_size=config['sae']['batch_size'],
        learning_rate=config['sae']['learning_rate'],
        n_epochs=config['sae']['n_epochs']
    )

def create_dataloaders_from_config(config: Dict[str, Any]):
    """Create dataloaders from dictionary config."""
    return create_dataloaders(
        dataset_type=config['dataset']['dataset_type'],
        batch_size=config['training']['batch_size'],
        selected_classes=config['dataset']['selected_classes'],
        image_size=config['dataset']['image_size'],
        data_dir=config['data_dir']
    )


def run_training_phase(config: Dict[str, Any], results_dir: Optional[Path] = None) -> Path:
    """Run training phase with optional checkpoint saving for LLC analysis."""
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

    # Log relevant aspects of the configuration
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
    
    # Log hardware and environment information
    log("\n=== Hardware Information ===")
    device = config['device']
    log(f"\tDevice: {device}")
    if device.type == 'cuda':
        log(f"\tGPU: {torch.cuda.get_device_name(device)}")
        log(f"\tCUDA Version: {torch.version.cuda}")
        log(f"\tAvailable GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    # Log PyTorch version
    log(f"\tPyTorch Version: {torch.__version__}")
    
    # Log dataset size information
    log("\n=== Dataset Information ===")
    train_loader, val_loader = create_dataloaders_from_config(config)
    log(f"\tTrain loader size: {len(train_loader.dataset)}")
    log(f"\tValidation loader size: {len(val_loader.dataset)}")
    
    attack_config = create_attack_config(config)
    model_config = create_model_config(config)
    training_config = create_training_config(config)
    
    # Train models with different adversarial strengths
    log("\n=== Starting Model Training ===")

    # log model structure and parameters
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
            model = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model_config=model_config,
                training_config=training_config,
                epsilon=epsilon,
                attack_config=attack_config,
                device=config['device'],
                logger=logger,
                save_dir=run_dir # Pass save directory for checkpoints
            )

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
                json.dump(robustness_results, f, indent=4, default=json_serializer)
            
    return results_dir


def run_sae_analysis(model, train_loader, adv_loader, layer_names, sae_config, run_dir, logger, results_entry):
    for layer_name in layer_names:
        logger.info(f"SAE evaluation for layer: {layer_name}")
        logger.info(f"Clean data")
        clean_metrics = measure_superposition(
            model=model,
            dataloader=train_loader,
            layer_name=layer_name,
            sae_config=sae_config,
            save_dir=run_dir / "saes_clean",
            logger=logger
        )
        logger.info(f"Adversarial data")
        adv_metrics = measure_superposition(
            model=model,
            dataloader=adv_loader,
            layer_name=layer_name,
            sae_config=sae_config,
            save_dir=run_dir / "saes_adv",
            logger=logger
        )
        results_entry['layers'][layer_name]['clean_feature_count'].append(clean_metrics['feature_count'])
        results_entry['layers'][layer_name]['adv_feature_count'].append(adv_metrics['feature_count'])
        logger.info(f"\tClean features: {clean_metrics['feature_count']:.2f}")
        logger.info(f"\tAdversarial features: {adv_metrics['feature_count']:.2f}")
        logger.info(f"\tFeature ratio (adv/clean): {adv_metrics['feature_count']/clean_metrics['feature_count']:.2f}")


def run_sae_checkpoint_analysis(checkpoint_dir, train_loader, sae_config, layer_names, epsilon, device, logger):
    return analyze_checkpoints_with_sae(
        checkpoint_dir=checkpoint_dir,
        train_loader=train_loader,
        sae_config=sae_config,
        layer_names=layer_names,
        epsilons_to_test=[0.0, epsilon],
        device=device,
        logger=logger
    )

def run_llc_checkpoint_analysis(checkpoint_dir, train_loader, llc_config, epsilon, device, logger):
    return analyze_checkpoints_with_llc(
        checkpoint_dir=checkpoint_dir,
        train_loader=train_loader,
        llc_config=llc_config,
        epsilons_to_test=[0.0, epsilon],
        device=device,
        logger=logger
    )

def run_combined_sae_llc_checkpoint_analysis(checkpoint_dir, train_loader, sae_config, llc_config, layer_names, epsilon, device, logger):
    return analyze_checkpoints_combined_sae_llc(
        checkpoint_dir=checkpoint_dir,
        train_loader=train_loader,
        sae_config=sae_config,
        llc_config=llc_config,
        layer_names=layer_names,
        epsilons_to_test=[0.0, epsilon],
        device=device,
        logger=logger
    )


def run_inference_time_llc(model, train_loader, adv_loader, attack_config, config, run_dir, logger, epsilon, llc_tuning_results):
    inference_dataloaders = {
        'clean': train_loader,
        f'eps_{epsilon}': adv_loader
    }
    # Add additional epsilon conditions for comprehensive analysis
    for test_eps in [0.05, 0.1, 0.2] if epsilon > 0 else [0.1, 0.2]:
        if test_eps != epsilon:
            test_loader = get_adversarial_dataloader(model, train_loader, attack_config, test_eps)
            inference_dataloaders[f'eps_{test_eps}'] = test_loader
    # Use tuned LLC hyperparameters if available
    llc_config = config['llc'].copy()
    if llc_tuning_results is not None:
        llc_config['epsilon'] = llc_tuning_results.get('recommended_epsilon', llc_config.get('epsilon'))
        llc_config['nbeta'] = llc_tuning_results.get('recommended_beta', llc_config.get('nbeta'))
    return measure_inference_time_llc(
        model=model,
        dataloaders=inference_dataloaders,
        llc_config=llc_config,
        save_dir=run_dir / "inference_llc",
        logger=logger
    )

def run_evaluation_phase(
    search_string: Optional[str] = None,
    results_dir: Optional[Path] = None,
    dataset_type: Optional[str] = None,
    enable_llc: bool = True,
    enable_sae: bool = True,
    enable_inference_llc: bool = False,
    enable_sae_checkpoints: bool = False
) -> Path:
    config = get_default_config(dataset_type=dataset_type)
    base_dir = Path(config['base_dir'])
    config, results_dir = get_config_and_results_dir(base_dir=base_dir, results_dir=results_dir, search_string=search_string)
    logger = setup_logger(results_dir)
    log = logger.info

    log("\n=== Starting Evaluation Phase ===")
    log(f"Results directory: {results_dir}")
    log(f"SAE analysis: {'enabled' if enable_sae else 'disabled'}")
    log(f"SAE checkpoint analysis: {'enabled' if enable_sae_checkpoints else 'disabled'}")
    log(f"LLC analysis: {'enabled' if enable_llc else 'disabled'}")
    log(f"Inference-time LLC analysis: {'enabled' if enable_inference_llc else 'disabled'}")

    train_loader, _ = create_dataloaders_from_config(config)
    attack_config = create_attack_config(config)
    sae_config = create_sae_config(config) if enable_sae or enable_sae_checkpoints else None

    results = {}

    for epsilon in config['adversarial']['train_epsilons']:
        eps_dir = results_dir / f"eps_{epsilon}"
        results[str(epsilon)] = {
            'robustness_score': [],
            'layers': {},
            'llc_clean': [],
            'llc_adversarial': [],
            'inference_llc': {},
            'sae_checkpoint_analysis': {}
        }
        layer_names = get_layer_names_for_model(config['model']['model_type']) if (enable_sae or enable_sae_checkpoints) else []

        if enable_sae:
            for layer_name in layer_names:
                results[str(epsilon)]['layers'][layer_name] = {
                    'clean_feature_count': [],
                    'adv_feature_count': []
                }

        for run_idx in range(config['adversarial']['n_runs']):
            run_dir = eps_dir / f"run_{run_idx}"
            log(f"\n--- Evaluating with epsilon={epsilon}, run {run_idx+1}/{config['adversarial']['n_runs']} ---")
            model, model_config = load_model(model_path=run_dir / "model.pt", device=config['device'])
            adv_loader = get_adversarial_dataloader(model, train_loader, attack_config, epsilon)

            # Load and store robustness results
            with open(run_dir / "robustness.json", 'r') as f:
                robustness = json.load(f)
            results[str(epsilon)]['robustness_score'].append(
                sum(float(v) for v in robustness.values()) / len(robustness)
            )
            if 'detailed_robustness' not in results[str(epsilon)]:
                results[str(epsilon)]['detailed_robustness'] = {}
            for test_eps, score in robustness.items():
                if test_eps not in results[str(epsilon)]['detailed_robustness']:
                    results[str(epsilon)]['detailed_robustness'][test_eps] = []
                results[str(epsilon)]['detailed_robustness'][test_eps].append(float(score))

            # 1. SAE analysis (final model)
            if enable_sae:
                run_sae_analysis(model, train_loader, adv_loader, layer_names, sae_config, run_dir, logger, results[str(epsilon)])

            # 2. Checkpoint analyses (SAE, LLC, or combined)
            checkpoint_dir = run_dir / "checkpoints"
            llc_tuning_results = None
            if checkpoint_dir.exists():
                if enable_llc and enable_sae_checkpoints:
                    log("Performing combined SAE + LLC checkpoint analysis...")
                    combined_results = run_combined_sae_llc_checkpoint_analysis(
                        checkpoint_dir, train_loader, sae_config, config['llc'], layer_names, epsilon, config['device'], logger
                    )
                    results[str(epsilon)].setdefault('combined_checkpoint_analysis', []).append(combined_results)
                    # Extract LLC tuning results for inference-time LLC
                    llc_tuning_results = combined_results['llc_analysis'].get('tuning_results', None)
                elif enable_llc:
                    log("Performing LLC checkpoint analysis...")
                    llc_results = run_llc_checkpoint_analysis(
                        checkpoint_dir, train_loader, config['llc'], epsilon, config['device'], logger
                    )
                    results[str(epsilon)].setdefault('checkpoint_llc_analysis', []).append(llc_results)
                    llc_tuning_results = llc_results.get('tuning_results', None)
                elif enable_sae_checkpoints:
                    log("Performing SAE checkpoint analysis...")
                    sae_checkpoint_results = run_sae_checkpoint_analysis(
                        checkpoint_dir, train_loader, sae_config, layer_names, epsilon, config['device'], logger
                    )
                    results[str(epsilon)].setdefault('sae_checkpoint_analysis', []).append(sae_checkpoint_results)

            # 3. Inference-time LLC (using tuned LLC hyperparameters)
            if enable_inference_llc:
                log("Performing inference-time LLC analysis...")
                inference_llc_results = run_inference_time_llc(
                    model, train_loader, adv_loader, attack_config, config, run_dir, logger, epsilon, llc_tuning_results
                )
                if inference_llc_results:
                    results[str(epsilon)]['inference_llc'][f'run_{run_idx}'] = inference_llc_results
                    log(f"\t  Inference LLC analysis completed")

    # Save results
    results_file = results_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=json_serializer)
    return results_dir


def get_layer_names_for_model(model_type: str) -> List[str]:
    """Determine layer names based on model type.
    
    Args:
        model_type: Type of model ('mlp', 'cnn', 'resnet18')
        
    Returns:
        List of layer names to analyze
    """
    model_type = model_type.lower()
    if model_type == 'mlp':  # for MNIST
        layer_names = [
            'relu1', # post-ReLU after each fc
            'relu2', 
            'relu3',
        ]
    elif model_type == 'simplemlp':
        layer_names = [
            'relu',
        ]
    elif model_type == 'cnn': # for MNIST
        layer_names = [
            'conv1.1', # post-ReLU after first conv (28×28)
            'conv2.1', # post-ReLU after second conv (14×14)
            'fc1.1', # post-ReLU after first fc 
        ]
    elif model_type == 'simplecnn':
        layer_names = [
            'relu',
            'fc',
        ]
    elif model_type == 'resnet18': # for CIFAR-10
        # Input (3×32×32) 
        # → Initial Processing (conv1, bn1, relu, maxpool)
        # → Stage 1: layer1 (2 BasicBlocks, 64 channels, 32×32)
        # → Stage 2: layer2 (2 BasicBlocks, 128 channels, 16×16) 
        # → Stage 3: layer3 (2 BasicBlocks, 256 channels, 8×8)
        # → Stage 4: layer4 (2 BasicBlocks, 512 channels, 4×4)
        # → Global pooling (avgpool) + classification (fc)
        layer_names = [
            'relu',           # early_features: edges/textures (32×32) (post-ReLU)
            # 'layer1.1.bn2',   # low_level: basic patterns (32×32) (pre-ReLU, residual)
            'layer1.1',       # low_level: basic patterns (32×32) (post-ReLU, combines skip and residual)
            # 'layer2.1.bn2',   # mid_level: object parts (16×16) (pre-ReLU, residual)
            'layer2.1',       # mid_level: object parts (16×16) (post-ReLU, combines skip and residual)
            # 'layer3.1.bn2',   # high_level: object concepts (8×8) (pre-ReLU, residual)
            'layer3.1',       # high_level: object concepts (8×8) (post-ReLU, combines skip and residual)
            # 'layer4.1.bn2',   # semantic: class-relevant features (4×4) (pre-ReLU, residual)
            'layer4.1',       # semantic: class-relevant features (4×4) (post-ReLU, combines skip and residual)
            'avgpool'         # Global features (512ch, 1×1)
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types are 'mlp', 'cnn', and 'resnet18'.")
    
    return layer_names

def run_analysis_phase(
    search_string: Optional[str] = None,
    results_dir: Optional[Path] = None,
    dataset_type: Optional[str] = None,
    plot_args: Optional[Dict[str, Any]] = None
) -> Path:
    """Run analysis phase with direct consumption of evaluation results.
    
    Args:
        search_string: Optional search string to find results directory
        results_dir: Optional path to results directory
        dataset_type: Optional dataset type to use for configuration
        
    Returns:
        Path to results directory
    """
    if not plot_args:
        plot_args = {
            'plot_legend': True,
            'plot_adversarial': True,
            'legend_alpha': 0.7,
            'plot_feature_counts': True,
            'y_ticks': [0.5, 1.0, 2.0],
            'y_tick_labels': ['0.5', '1', '2'],
            'plot_robustness': True,
            'plot_normalized_feature_counts': True,
        }
    
    config = get_default_config(dataset_type=dataset_type)
    config, results_dir = get_config_and_results_dir(base_dir=Path(config['base_dir']), results_dir=results_dir, search_string=search_string)
    
    # Set up logger
    logger = setup_logger(results_dir)
    
    logger.info("\n=== Starting Analysis Phase ===")
    logger.info(f"Using results directory: {results_dir}")
    
    # Load evaluation results directly
    results_file = results_dir / "results.json"
    if not results_file.exists():
        # raise FileNotFoundError(f"Results file not found: {results_file}")
        print(f"Results file not found: {results_file}")
        return results_dir
    
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Only include layers specified by get_layer_names based on model type
    model_type = config['model']['model_type']
    layer_names = get_layer_names_for_model(model_type)
    # print(f"Layer names: {layer_names}")
    # Extract available layers from results
    available_layers = next(iter(results.values())).get('layers', {}).keys()
    
    # Filter results to only include the specified layers
    for epsilon_key in results:
        if 'layers' in results[epsilon_key]:
            filtered_layers = {}
            for layer_name in layer_names:
                if layer_name in results[epsilon_key]['layers']:
                    filtered_layers[layer_name] = results[epsilon_key]['layers'][layer_name]
            results[epsilon_key]['layers'] = filtered_layers
    
    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    generate_plots(results, plots_dir, results_dir, plot_args)
    
    # *** LLC-specific analysis plots ***
    epsilons = sorted([float(eps) for eps in results.keys()])
    
    # Extract LLC data for multi-epsilon plots
    llc_checkpoint_evolution = {}

    # Check if LLC data is available
    has_llc_data = any('checkpoint_llc_analysis' in results[str(eps)] 
                       for eps in epsilons)

    # Check if SAE checkpoint data is available
    has_sae_checkpoint_data = any('sae_checkpoint_analysis' in results[str(eps)]
                                 for eps in epsilons)

    if has_llc_data:
        logger.info("Generating LLC-specific analysis plots...")
        
        for eps in epsilons:
            eps_str = str(eps)
            if 'checkpoint_llc_analysis' in results[eps_str]:
                # Extract checkpoint evolution data
                llc_values = []
                for checkpoint_analysis in results[eps_str]['checkpoint_llc_analysis']:
                    if 'epsilon_analysis' in checkpoint_analysis:
                        # Use adversarial data (eps) for evolution
                        if eps_str in checkpoint_analysis['epsilon_analysis']:
                            for checkpoint_data in checkpoint_analysis['epsilon_analysis'][eps_str]:
                                llc_values.append(checkpoint_data['llc_mean'])
                
                if llc_values:
                    llc_checkpoint_evolution[eps] = llc_values

    # *** NEW: Combined LLC and SAE CHECKPOINT EVOLUTION ANALYSIS ***
    has_combined_checkpoint_data = any('combined_checkpoint_analysis' in results[str(eps)] 
                                      for eps in epsilons 
                                      if 'combined_checkpoint_analysis' in results[str(eps)])
    
    if has_combined_checkpoint_data:
        logger.info("Generating combined SAE + LLC checkpoint evolution plots...")
        
        # Get model metadata
        model_type = config['model']['model_type']
        n_classes = len(config['dataset']['selected_classes'])
        layer_names = get_layer_names_for_model(model_type)
        
        # Aggregate data across runs for each epsilon and layer
        for eps in epsilons:
            eps_str = str(eps)
            if 'combined_checkpoint_analysis' in results[eps_str]:
                
                # For each layer, create aggregated evolution plots
                for layer_name in layer_names:
                    
                    # Collect LLC and SAE data across all runs
                    aggregated_llc_data = []
                    aggregated_sae_data = []
                    
                    for run_analysis in results[eps_str]['combined_checkpoint_analysis']:
                        # Extract LLC data
                        if ('llc_analysis' in run_analysis and 
                            'epsilon_analysis' in run_analysis['llc_analysis'] and
                            eps_str in run_analysis['llc_analysis']['epsilon_analysis']):
                            
                            llc_data = run_analysis['llc_analysis']['epsilon_analysis'][eps_str]
                            aggregated_llc_data.extend(llc_data)
                        
                        # Extract SAE data
                        if ('sae_analysis' in run_analysis and
                            'epsilon_analysis' in run_analysis['sae_analysis'] and
                            eps_str in run_analysis['sae_analysis']['epsilon_analysis'] and
                            layer_name in run_analysis['sae_analysis']['epsilon_analysis'][eps_str]):
                            
                            sae_data = run_analysis['sae_analysis']['epsilon_analysis'][eps_str][layer_name]
                            aggregated_sae_data.extend(sae_data)
                    
                    # Create aggregated evolution plot
                    if aggregated_llc_data and aggregated_sae_data:
                        # fig = plot_aggregated_llc_sae_evolution(
                        #     llc_data=aggregated_llc_data,
                        #     sae_data=aggregated_sae_data,
                        #     layer_name=layer_name,
                        #     epsilon=eps,
                        #     n_runs=config['adversarial']['n_runs'],
                        #     title=f"Aggregated LLC vs SAE Evolution - ε={eps}, {layer_name} - {model_type.upper()} {n_classes}-class",
                        #     save_path=plots_dir / f"aggregated_llc_sae_evolution_{layer_name.replace('.', '_')}_eps_{eps}.png"
                        # )

                        # Also: Dual-axis aggregated plot
                        fig_dual = plot_llc_sae_dual_axis_evolution(
                            llc_data=aggregated_llc_data,
                            sae_data=aggregated_sae_data,
                            layer_name=layer_name,
                            epsilon=eps,
                            n_runs=config['adversarial']['n_runs'],
                            title=f"Aggregated LLC vs SAE Dual Axis - ε={eps}, {layer_name} - {model_type.upper()} {n_classes}-class",
                            save_path=plots_dir / f"aggregated_llc_sae_dual_axis_{layer_name.replace('.', '_')}_eps_{eps}.pdf"
                        )
    
    if has_llc_data and has_sae_checkpoint_data:
        logger.info("Generating combined LLC and SAE analysis plots...")
        
        # Get layer names from SAE results
        layer_names = list(next(iter(results.values()))['layers'].keys())
        
        for eps in epsilons:
            eps_str = str(eps)
            
            # Skip if we don't have both types of data
            if not (('checkpoint_llc_analysis' in results[eps_str]) and 
                   ('sae_checkpoint_analysis' in results[eps_str])):
                continue
            
            # Process each layer
            for layer_name in layer_names:
                logger.info(f"Generating combined plots for ε={eps}, layer={layer_name}")
                
                try:
                    # 1. Side-by-side evolution plot
                    plot_combined_llc_sae_evolution(
                        llc_checkpoint_results=results[eps_str]['checkpoint_llc_analysis'][0],  # Use first run
                        sae_checkpoint_results=results[eps_str]['sae_checkpoint_analysis']['run_0'],  # Use first run
                        layer_name=layer_name,
                        epsilon=eps,
                        title=f"LLC vs SAE Evolution - ε={eps}, {layer_name}",
                        save_path=plots_dir / f"llc_sae_evolution_eps_{eps}_{layer_name.replace('.', '_')}.png"
                    )
                    
                    # 2. Correlation over time plot
                    plot_llc_vs_sae_correlation_over_time(
                        llc_checkpoint_results=results[eps_str]['checkpoint_llc_analysis'][0],  # Use first run
                        sae_checkpoint_results=results[eps_str]['sae_checkpoint_analysis']['run_0'],  # Use first run
                        layer_name=layer_name,
                        epsilon=eps,
                        title=f"LLC vs SAE Correlation - ε={eps}, {layer_name}",
                        save_path=plots_dir / f"llc_sae_correlation_time_eps_{eps}_{layer_name.replace('.', '_')}.png"
                    )
                    
                except Exception as e:
                    logger.error(f"Error generating combined plots for ε={eps}, layer={layer_name}: {e}")
    
    # 2. Generate clean vs adversarial LLC comparison
    if any('llc_clean' in results[str(eps)] and 'llc_adversarial' in results[str(eps)] 
           for eps in epsilons):
        
        clean_llc_data = {}
        adv_llc_data = {}
        
        for eps in epsilons:
            eps_str = str(eps)
            
            # Extract clean LLC data (final values)
            if 'llc_clean' in results[eps_str] and results[eps_str]['llc_clean']:
                clean_llc_data[eps] = [entry['llc_mean'] for entry in results[eps_str]['llc_clean'] 
                                      if entry['llc_mean'] is not None]
            
            # Extract adversarial LLC data (final values)
            if 'llc_adversarial' in results[eps_str] and results[eps_str]['llc_adversarial']:
                adv_llc_data[eps] = [entry['llc_mean'] for entry in results[eps_str]['llc_adversarial'] 
                                    if entry['llc_mean'] is not None]
        
        if clean_llc_data and adv_llc_data:
            plot_clean_vs_adversarial_llc(
                clean_llc=clean_llc_data,
                adv_llc=adv_llc_data,
                title="Clean vs Adversarial LLC Analysis",
                save_path=plots_dir / "clean_vs_adversarial_llc.png"
            )
    
    # 3. Combined SAE + LLC correlation plots (if both are available)
    has_sae_data = any('layers' in results[str(eps)] and 
                      any('clean_feature_count' in results[str(eps)]['layers'][layer] 
                          for layer in results[str(eps)]['layers'])
                      for eps in epsilons)
    
    if has_sae_data and llc_checkpoint_evolution:
        logger.info("Generating combined SAE + LLC correlation plots...")
        
        # Get model metadata
        model_type = config['model']['model_type']
        n_classes = len(config['dataset']['selected_classes'])
        
        # Extract SAE data for correlation
        layers = list(results[str(epsilons[0])]['layers'].keys())
        
        for layer_name in layers:
            sae_data = {}
            llc_data = {}
            
            for eps in epsilons:
                eps_str = str(eps)
                
                # Get SAE feature counts
                if ('layers' in results[eps_str] and 
                    layer_name in results[eps_str]['layers'] and
                    'clean_feature_count' in results[eps_str]['layers'][layer_name]):
                    
                    feature_counts = results[eps_str]['layers'][layer_name]['clean_feature_count']
                    sae_data[eps] = [x for x in feature_counts if x is not None]
                
                # Get LLC data (use final values)
                if eps in llc_checkpoint_evolution:
                    # Use the final LLC value for correlation
                    final_llc = llc_checkpoint_evolution[eps][-1] if llc_checkpoint_evolution[eps] else None
                    if final_llc is not None:
                        llc_data[eps] = [final_llc] * len(sae_data.get(eps, [1]))
            
            if sae_data and llc_data:
                plot_combined_sae_llc_correlation(
                    sae_data=sae_data,
                    llc_data=llc_data,
                    title=f"SAE vs LLC Correlation - {layer_name} - {model_type.upper()} {n_classes}-class",
                    save_path=plots_dir / f"sae_llc_correlation_{layer_name.replace('.', '_')}.png"
                )

    # NEW: *** INFERENCE-TIME LLC ANALYSIS PLOTS ***
    has_inference_llc_data = any('inference_llc' in results[str(eps)] and results[str(eps)]['inference_llc']
                                for eps in epsilons)
    
    if has_inference_llc_data:
        logger.info("Generating inference-time LLC analysis plots...")
        
        # Aggregate inference LLC data across runs
        aggregated_inference_llc = {}
        
        for eps in epsilons:
            eps_str = str(eps)
            if 'inference_llc' in results[eps_str]:
                # Collect data from all runs
                for run_key, run_data in results[eps_str]['inference_llc'].items():
                    for condition, llc_data in run_data.items():
                        # Extract epsilon value from condition name
                        if condition == 'clean':
                            condition_eps = 0.0
                        else:
                            try:
                                condition_eps = float(condition.split('_')[-1])
                            except:
                                condition_eps = 0.0
                        
                        if condition_eps not in aggregated_inference_llc:
                            aggregated_inference_llc[condition_eps] = {
                                'means': [],
                                'traces': []
                            }
                        
                        # Collect means and traces
                        aggregated_inference_llc[condition_eps]['means'].append(llc_data['mean'])
                        if 'trace' in llc_data and llc_data['trace'] is not None:
                            aggregated_inference_llc[condition_eps]['traces'].append(llc_data['trace'])
        
        # Generate inference-time plots if we have data
        if aggregated_inference_llc:
            # Convert to the format expected by plotting functions
            inference_plot_data = {}
            for eps, data in aggregated_inference_llc.items():
                inference_plot_data[eps] = {
                    'means': np.array(data['means']),
                    'mean': np.mean(data['means']),
                    'std': np.std(data['means']),
                    'trace': data['traces'][0] if data['traces'] else None  # Use first available trace
                }
            
            # 1. Plot LLC traces
            plot_llc_traces(
                llc_results=inference_plot_data,
                save_path=plots_dir / "inference_llc_traces.pdf",
                show=False
            )
            
            # 2. Plot LLC distribution
            plot_llc_distribution(
                llc_results=inference_plot_data,
                save_path=plots_dir / "inference_llc_distribution.pdf",
                show=False
            )
            
            # 3. Plot LLC mean vs std
            plot_llc_mean_std(
                llc_results=inference_plot_data,
                save_path=plots_dir / "inference_llc_mean_std.pdf",
                show=False
            )
            
            logger.info("Inference-time LLC plots generated successfully")
    
    logger.info(f"\n=== Analysis Phase Completed ===")
    logger.info(f"Results saved to: {plots_dir}")
    
    return results_dir


def run_model_class_experiment(
    model_types: List[str],
    class_counts: List[int],
    dataset_types: List[str],
    attack_types: List[str],
    enable_llc: bool = True,
    enable_sae: bool = True,
    enable_inference_llc: bool = False,  # Control inference-time LLC
    enable_sae_checkpoints: bool = False,  # Enable SAE checkpoint analysis
    hidden_dims: Optional[List[int]] = None,
    testing_mode: bool = False
):
    """Run experiments across model types, class counts, and attack types.
    
    Args:
        model_types: List of model types to test
        class_counts: List of class counts to test
        dataset_types: List of dataset types to test
        attack_types: List of attack types to test
        testing_mode: Whether to run in testing mode
        enable_llc: Whether to enable LLC analysis
        enable_sae: Whether to enable SAE analysis
        enable_inference_llc: Whether to enable inference-time LLC analysis
        enable_sae_checkpoints: Whether to enable SAE checkpoint analysis
    """
    for dataset_type in dataset_types:
        
        for attack_type in attack_types:
            for model_type in model_types:
                for n_classes in class_counts:
                    config = get_default_config(testing_mode=testing_mode, dataset_type=dataset_type, selected_classes=tuple(range(n_classes)))
                    text = f"Running experiment: {model_type.upper()} with {n_classes} classes on {dataset_type.upper()} using {attack_type.upper()}"
                    if not enable_sae:
                        text += " (SAE disabled)"
                    if not enable_llc:
                        text += " (LLC disabled)"
                    if enable_inference_llc:
                        text += " (Inference LLC enabled)"
                    
                    print(f"\n{'='*len(text)}")
                    print(text)
                    print(f"{'='*len(text)}\n")

                    # Use provided hidden_dims if available, otherwise use defaults
                    if hidden_dims:
                        # Run experiments for each hidden dimension
                        dims_to_test = hidden_dims
                    else:
                        # Use default hidden_dim based on model type
                        if model_type == 'cnn':
                            dims_to_test = [16]
                        elif model_type == 'mlp':
                            dims_to_test = [32]
                        elif model_type == 'simplemlp':
                            dims_to_test = [128]
                        elif model_type == 'simplecnn':
                            dims_to_test = [8]
                        else:
                            dims_to_test = [32]  # Default fallback
                    
                    for hidden_dim in dims_to_test:
                        dim_text = f"Training with hidden_dim={hidden_dim}"
                        print(f"\n{'-'*len(dim_text)}")
                        print(dim_text)
                        print(f"{'-'*len(dim_text)}\n")
                        
                        # Update config with current hidden_dim and other settings
                        config = update_config(config, {
                            'model': {
                                'model_type': model_type,
                                'hidden_dim': hidden_dim
                            },
                            'adversarial': {'attack_type': attack_type},
                            'dataset': {'selected_classes': tuple(range(n_classes))},
                        })
                        config = update_config(config, {
                            'comment': f"h{hidden_dim}"
                        })

                    results_dir = run_training_phase(config)
                    try: 
                        run_evaluation_phase(
                            results_dir=results_dir, 
                            dataset_type=dataset_type, 
                            enable_llc=enable_llc, 
                            enable_sae=enable_sae,
                            enable_sae_checkpoints=enable_sae_checkpoints,
                            enable_inference_llc=enable_inference_llc
                        )
                    except Exception as e:
                        print(f"Error running evaluation phase: {e}")
                    try:
                        run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)
                    except Exception as e:
                        print(f"Error running analysis phase: {e}")

 
def quick_test(
    model_type: str = None, 
    dataset_type: str = "cifar10", 
    testing_mode: bool = True, 
    enable_llc: bool = True,
    enable_sae: bool = True,
    enable_inference_llc: bool = False,
    enable_sae_checkpoints: bool = False  # NEW: Enable SAE checkpoint analysis
):
    """Run a quick test with standard adversarial training configuration.
    
    Args:
        model_type: Optional model type override (default: None, uses standard for dataset)
        dataset_type: Dataset type ("cifar10" or "mnist")
        testing_mode: Whether to use minimal settings for testing
    """
    # Quick test configuration
    config = get_default_config(dataset_type=dataset_type, testing_mode=testing_mode, selected_classes=(0, 1))
    
    # Run a single training experiment with minimal settings
    print(f"Running quick test on {dataset_type} with {config['model']['model_type']}...")
    
    # Customize config for quick test if model_type provided
    if model_type is not None:
        config = update_config(config, {
            'model': {'model_type': model_type}
        })
    
    # Add comment
    comment_parts = [f'quick_test_{dataset_type}_{config["model"]["model_type"]}_with_llc_sae']
    if enable_inference_llc:
        comment_parts.append('inference_llc')
    
    config = update_config(config, {
        'comment': '_'.join(comment_parts)
    })

    # Print key configuration
    print(f"Model: {config['model']['model_type']}")
    print(f"Dataset: {config['dataset']['dataset_type']}")
    print(f"Attack: {config['adversarial']['attack_type']} with {config['adversarial']['pgd_steps']} steps")
    print(f"Epsilon: {config['adversarial']['train_epsilons']}")
    print(f"Epochs: {config['training']['n_epochs']}")
    print(f"LLC checkpointing: every {config['llc']['measure_frequency']} epochs")
    print(f"Inference-time LLC: {'enabled' if enable_inference_llc else 'disabled'}")
    print(f"SAE checkpoint analysis: {'enabled' if enable_sae_checkpoints else 'disabled'}")

    # Run training phase
    results_dir = run_training_phase(config)
    
    # Run evaluation on the trained model (with both SAE and LLC)
    run_evaluation_phase(
        results_dir=results_dir, 
        dataset_type=dataset_type, 
        enable_llc=enable_llc, 
        enable_sae=enable_sae,
        enable_inference_llc=enable_inference_llc,
        enable_sae_checkpoints=enable_sae_checkpoints  # Enable SAE checkpoint analysis
    )
    
    # Generate analysis plots
    run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)
    
    print(f"Quick test completed. Results in: {results_dir}")
    return results_dir

def reanalyze_results(
    results_dir: Optional[str] = None,
    search_string: Optional[str] = None,
    dataset_type: Optional[str] = None,
    plot_args: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """Reanalyze existing results without re-running training or evaluation."""
    # Convert results_dir to Path object if provided
    if results_dir is not None:
        results_dir = Path(results_dir)
    
    # If dataset_type not provided but results_dir is, try to extract it from path
    if dataset_type is None and results_dir is not None:
        # Look for known dataset types in the path
        known_datasets = ['mnist', 'cifar10', 'cifar100']
        for dataset in known_datasets:
            if dataset in str(results_dir).lower():
                dataset_type = dataset
                break
        if dataset_type is None:
            raise ValueError(f"Could not determine dataset_type from results_dir. Please provide dataset_type explicitly.")
    
    # Get config and results directory
    config = get_default_config(dataset_type=dataset_type)
    base_dir = Path(config['base_dir'])
    
    # If results_dir is provided, ensure it exists
    if results_dir is not None and not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    # Get config and results directory
    config, results_dir = get_config_and_results_dir(
        base_dir=base_dir, 
        results_dir=results_dir,  # Now it's already a Path object
        search_string=search_string
    )
    
    # Convert output_dir to Path if provided
    if output_dir is not None:
        output_dir = Path(output_dir)
    
    # Set up logger - results_dir is now guaranteed to be a Path object
    logger = setup_logger(results_dir)
    log = logger.info
    
    log(f"\n=== Starting Reanalysis of Results ===")
    log(f"Loading results from: {results_dir}")
    log(f"Dataset type: {dataset_type}")
    
    # Load and analyze results
    from analysis import load_and_analyze_existing_results
    figures = load_and_analyze_existing_results(
        results_dir=results_dir,
        plot_args=plot_args,
        output_dir=output_dir
    )
    
    log(f"Generated {len(figures)} plots")
    if output_dir:
        log(f"Plots saved to: {output_dir}")
    else:
        log(f"Plots saved to: {results_dir}/plots")
    
    return figures

# %%
if __name__ == "__main__":
    """
    Example usage - Quick test with both SAE, LLC, and optional inference-time LLC
    """
    # quick_test(
    #     model_type='resnet18', 
    #     dataset_type="mnist", 
    #     testing_mode=True, 
    #     enable_sae=True,
    #     enable_llc=True,
    #     enable_inference_llc=True,
    #     enable_sae_checkpoints=True  
    # )

    """
    reanalyze results
    """
    plot_args = {
        'plot_legend': True,
        'plot_adversarial': True,
        'legend_alpha': 1.0,
        'plot_feature_counts': False,
        'plot_normalized_feature_counts': False,
        'plot_robustness': True,  # Make sure robustness plot is enabled
    }
    
    figures = reanalyze_results(
        results_dir="../../results/adversarial_robustness/mnist/2025-06-19_13-01_simplemlp_2-class_pgd_quick_test_mnist_simplemlp_with_llc_sae_inference_llc",
        plot_args=plot_args
    )


    '''Example usage - Full experiment with all analyses'''
    # run_model_class_experiment(
    #     model_types=['cnn', 'mlp'],
    #     class_counts=[2, 3],
    #     dataset_types=['mnist'],
    #     attack_types=['pgd'],
    #     testing_mode=False,
    #     enable_llc=True,
    #     enable_sae=True,
    #     enable_sae_checkpoints=True,  # Enable SAE checkpoint analysis
    #     enable_inference_llc=True  # Enable inference-time LLC for detailed analysis
    # )



    '''analyze cifar10 cnn 2-class pgd'''
    '''run evaluation for resnet18 2, 3, 5, 10 class pgd, and fgsm and analyze'''
    # model_types = ['resnet18']
    # class_counts = [2, 3, 5, 10]
    # dataset_types = ['cifar10']
    # attack_types = ['pgd', 'fgsm']
    # # run_model_class_experiment(model_types, class_counts, dataset_types, attack_types, testing_mode=False)
    # for model_type in model_types:
    #     for class_count in class_counts:
    #         for dataset_type in dataset_types:
    #             for attack_type in attack_types:
    #                 run_evaluation_phase(search_string=f"{model_type}_{class_count}-class_{attack_type}", dataset_type=dataset_type)
    #                 run_analysis_phase(search_string=f"{model_type}_{class_count}-class_{attack_type}", dataset_type=dataset_type)


    # run a single experiment with simplemlp 
    # model_types = ['simplemlp', 'simplecnn']
    # class_counts = [3, 5]
    # dataset_types = ['mnist']
    # attack_types = ['fgsm', 'pgd']
    # testing_mode = False
    # run_model_class_experiment(model_types, class_counts, dataset_types, attack_types, testing_mode)
    # run_evaluation_phase(search_string="simplemlp_2-class_fgsm", dataset_type="mnist")

    # run mnist mlp 2-class pgd
    # model_types = ['mlp']
    # class_counts = [2] or [3, 5, 10]
    # dataset_types = ['mnist']
    # attack_types = ['pgd']
    # testing_mode = False
    # # run_model_class_experiment(model_types, class_counts, dataset_types, attack_types, testing_mode)
    # run_analysis_phase(search_string="43_mlp_2-class_pgd", dataset_type="mnist")
    # for hidden_dim in [8, 32]:
    #     run_analysis_phase(search_string=f"simplemlp_2-class_fgsm_h{hidden_dim}", dataset_type="mnist")
    # plot_args = {
    #     'plot_legend': False,
    #     'plot_adversarial': True,
    #     'legend_alpha': 1.0, 
    #     # 'y_lim': (0.35, 8.),
    #     'y_ticks': [0.5, 1.0, 2.0],
    #     'y_tick_labels': ['0.5', '1', '2'],
    #     'y_scale': 'log',
    #     'plot_feature_counts': False,
    #     'plot_robustness': False,
    #     'plot_normalized_feature_counts': True,
    # }
    # for model_type in ['simplemlp', 'simplecnn']:
    # for model_type in ['_mlp', '_cnn']: # 'simplemlp', 'simplecnn']:
    # dataset_type = 'cifar10'
    # dataset_type = 'mnist'
    # for model_type in ['_mlp', '_cnn']: # 'simplemlp', 'simplecnn']:
    # for model_type in ['simplemlp', 'simplecnn']: 
    #     for n_classes in [2, 10]:
    #         for attack_type in ['fgsm', 'pgd']:
    #             run_analysis_phase(search_string=f"{model_type}_{n_classes}-class_{attack_type}", dataset_type=dataset_type, plot_args=plot_args)
    # run_analysis_phase(search_string="simplecnn_5-class_pgd", dataset_type="mnist", plot_args=plot_args)



    # run_model_class_experiment(
    #     model_types=['simplemlp', 'simplecnn'],
    #     class_counts=[2, 3, 5, 10],
    #     dataset_types=['mnist'],
    #     attack_types=['pgd'],
    #     testing_mode=False
    # )
# %% 
    # MNIST experiment
    # config = get_default_config(
    #     dataset_type="cifar10", 
    #     testing_mode=True,
    #     selected_classes=(0, 1)
    # )
    # config = update_config(config, {
    #     'model': {'model_type': 'resnet18'},
    #     'adversarial': {'attack_type': 'fgsm', 'train_epsilons': [0.0, 0.1, 0.2], 'test_epsilons': [0.0, 0.1, 0.2], 'n_runs': 1},
    # })
    # results_dir = run_training_phase(config)
    # run_evaluation_phase(results_dir=results_dir, dataset_type="cifar10")
    # run_analysis_phase(results_dir=results_dir, dataset_type="cifar10")

    # === PREVIOUS EXPERIMENTS (commented out) ===
    # experiments to start:
    # - epsilon scaling and limits:
    # mnist cnn 2-class fgsm - 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 - 100 epochs, 1 run  

    # Run quick test
    # quick_test(model_type='resnet18', dataset_type="cifar10", testing_mode=True)
    # quick_test(model_type='cnn', dataset_type="mnist", testing_mode=True)
    # quick_test(model_type='mlp', dataset_type="cifar10", testing_mode=True)
    # quick_test(model_type='cnn', dataset_type="cifar10", testing_mode=True)
    # quick_test(model_type='resnet18', dataset_type="cifar10", testing_mode=True)
    # run_evaluation_phase(search_string="cnn_2-class", dataset_type="mnist")
    # run_analysis_phase(search_string="cnn_3-class", dataset_type="mnist")

    # Run single experiment: standard experiment
    # dataset_type = "mnist"
    # comment = "test"
    # config = get_default_config(testing_mode=True, dataset_type=dataset_type)
    # # update config for standard experiment
    # config = update_config(config, {
    #     'model': {
    #         'model_type': 'cnn',
    #         'hidden_dim': 32 
    #     },
    #     'dataset': {
    #         'selected_classes': tuple(i for i in range(10)) 
    #     },
    #     'training': {
    #         'n_epochs': 10
    #     },
    #     'adversarial': {
    #         'train_epsilons': [0.0, 0.1, 0.2],
    #         'test_epsilons': [0.0, 0.1, 0.2],
    #         'n_runs': 1,  # 1 run for quick test
    #         'attack_type': 'fgsm',
    #     },
    #     'comment': comment
    # })

    # results_dir = run_training_phase(config)
    # run_evaluation_phase(results_dir=results_dir, dataset_type=dataset_type)
    # run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)
    # run_analysis_phase(search_string="cnn_2-class", dataset_type="cifar10")

    # NOTE: to run
    # 2025-05-23 10-55
    # base_channels = [16, 32, 64, 128]
    # for base_channel in base_channels:
    #     config = get_default_config(testing_mode=False, dataset_type="mnist")
    #     config = update_config(config, {
    #         'model': {'model_type': 'cnn', 'hidden_dim': base_channel}
    #     })
    #     results_dir = run_training_phase(config)
    #     run_evaluation_phase(results_dir=results_dir, dataset_type=dataset_type)
    #     run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)

# %%
