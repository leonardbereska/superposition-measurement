# %%
"""Main script for running adversarial robustness experiments."""

import os
import torch
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import configuration, datasets, utils
from config import get_default_config, update_config
from datasets_adversarial import create_dataloaders
from utils import json_serializer, setup_results_dir, save_config, get_config_and_results_dir, setup_logger, create_directory

# Import components for different phases 
from training import train_model, evaluate_model_performance, load_model, save_model
from evaluation import measure_superposition
from analysis import generate_plots
from attacks import AttackConfig, generate_adversarial_examples, get_adversarial_dataloader
from models import ModelConfig
from sae import SAEConfig

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
    """Run training phase.
    
    Args:
        config: Experiment configuration
        results_dir: Optional custom results directory
        
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

    # Log relevant aspects of the configuration
    log("\n=== Configuration Summary ===")
    log(f"\tDataset: {config['dataset']['dataset_type']}")
    log(f"\tNumber of classes: {len(config['dataset']['selected_classes'])}")
    log(f"\tModel type: {config['model']['model_type'].upper()}")
    log(f"\tAdversarial attack: {config['adversarial']['attack_type'].upper()}")
    log(f"\tTraining epsilons: {config['adversarial']['train_epsilons']}")
    log(f"\tTest epsilons: {config['adversarial']['test_epsilons']}")
    log(f"\tNumber of runs: {config['adversarial']['n_runs']}")
    
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
    
    # Train models with different adversarial strengths
    log("\n=== Starting Model Training ===")
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
                n_epochs=config['training']['n_epochs'],
                learning_rate=config['training']['learning_rate'],
                epsilon=epsilon,
                attack_config=attack_config,
                device=config['device'],
                logger=logger
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

def run_evaluation_phase(
    search_string: Optional[str] = None,
    results_dir: Optional[Path] = None,
    dataset_type: Optional[str] = None
) -> Path:
    """Run evaluation phase with minimal saving overhead.
    
    Args:
        search_string: Search string to find results directory
        results_dir: Results directory
        dataset_type: Dataset type
    Returns:
        Path to results directory
    """
    config = get_default_config(dataset_type=dataset_type)
    base_dir = Path(config['base_dir'])
    config, results_dir = get_config_and_results_dir(base_dir=base_dir, results_dir=results_dir, search_string=search_string)

    # Set up logger
    logger = setup_logger(results_dir)
    log = logger.info
    
    log("\n=== Starting Evaluation Phase ===")
    log(f"Results directory: {results_dir}")
    
    # Log configuration summary
    log("\n=== Configuration Summary ===")
    log(f"\tDataset: {config['dataset']['dataset_type']}")
    log(f"\tNumber of classes: {len(config['dataset']['selected_classes'])}")
    log(f"\tModel type: {config['model']['model_type']}")
    log(f"\tAdversarial attack: {config['adversarial']['attack_type']}")
    log(f"\tTraining epsilons: {config['adversarial']['train_epsilons']}")
    log(f"\tTest epsilons: {config['adversarial']['test_epsilons']}")
    log(f"\tNumber of runs: {config['adversarial']['n_runs']}")
    
    # Create dataloaders and configs
    train_loader, _ = create_dataloaders_from_config(config)
    attack_config = create_attack_config(config)
    sae_config = create_sae_config(config)
    
    log("\n=== Dataset Information ===")
    log(f"\tTrain loader size: {len(train_loader.dataset)}")
    
    # Cleaner results structure - a simple nested dictionary
    results = {}
    
    # Evaluate each epsilon
    for epsilon in config['adversarial']['train_epsilons']:
        eps_dir = results_dir / f"eps_{epsilon}"
        
        # Initialize epsilon entry
        results[str(epsilon)] = {
            'robustness_score': [],
            'layers': {}  # Organize by layer
        }
        
        # Evaluate each run
        for run_idx in range(config['adversarial']['n_runs']):
            run_dir = eps_dir / f"run_{run_idx}"
            log(f"\n--- Evaluating with epsilon={epsilon}, run {run_idx+1}/{config['adversarial']['n_runs']} ---")
            
            # Load model
            model, model_config = load_model(
                model_path=run_dir / "model.pt",
                device=config['device']
            )
            
            # Get layer names for the model
            layer_names = get_layer_names_for_model(model_config.model_type)
            
            # Initialize layer entries if not present
            for layer_name in layer_names:
                if layer_name not in results[str(epsilon)]['layers']:
                    results[str(epsilon)]['layers'][layer_name] = {
                        'clean_feature_count': [],
                        'adv_feature_count': []
                    }
            
            # Create adversarial loader
            adv_loader = get_adversarial_dataloader(model, train_loader, attack_config, epsilon)
            
            # Load robustness results
            with open(run_dir / "robustness.json", 'r') as f:
                robustness = json.load(f)
            
            # Store robustness for each test epsilon
            results[str(epsilon)]['robustness_score'].append(
                sum(float(v) for v in robustness.values()) / len(robustness)
            )
            
            # Store detailed robustness for each test epsilon
            if 'detailed_robustness' not in results[str(epsilon)]:
                results[str(epsilon)]['detailed_robustness'] = {}
                
            for test_eps, score in robustness.items():
                if test_eps not in results[str(epsilon)]['detailed_robustness']:
                    results[str(epsilon)]['detailed_robustness'][test_eps] = []
                results[str(epsilon)]['detailed_robustness'][test_eps].append(float(score))
            
            # Evaluate each layer
            for layer_name in layer_names:
                log(f"Layer: {layer_name}")
                
                try:
                    # Measure feature organization for clean data
                    log(f"Clean data")
                    clean_metrics = measure_superposition(
                        model=model,
                        dataloader=train_loader,
                        layer_name=layer_name,
                        sae_config=sae_config,
                        save_dir=run_dir / "saes_clean",
                        logger=logger
                    )
                    
                    # Measure feature organization for adversarial data
                    log(f"Adversarial data")
                    adv_metrics = measure_superposition(
                        model=model,
                        dataloader=adv_loader,
                        layer_name=layer_name,
                        sae_config=sae_config,
                        save_dir=run_dir / "saes_adv",
                        logger=logger
                    )
                    
                    # Store results in cleaner nested structure
                    results[str(epsilon)]['layers'][layer_name]['clean_feature_count'].append(
                        clean_metrics['feature_count']
                    )
                    results[str(epsilon)]['layers'][layer_name]['adv_feature_count'].append(
                        adv_metrics['feature_count']
                    )
                    
                    log(f"\tClean features: {clean_metrics['feature_count']:.2f}")
                    log(f"\tAdversarial features: {adv_metrics['feature_count']:.2f}")
                    log(f"\tFeature ratio (adv/clean): {adv_metrics['feature_count']/clean_metrics['feature_count']:.2f}")
                    
                except Exception as e:
                    log(f"\tError evaluating layer {layer_name}: {e}")
                    # Add placeholder values to maintain consistency
                    results[str(epsilon)]['layers'][layer_name]['clean_feature_count'].append(None)
                    results[str(epsilon)]['layers'][layer_name]['adv_feature_count'].append(None)
    
    # Save results in a cleaner nested structure
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
    if model_type == 'mlp':
        layer_names = ['relu']  # Hidden activation 
    elif model_type == 'cnn':
        layer_names = ['features.1', 'features.4']  # ReLU after first and second conv
    elif model_type == 'resnet18':
        layer_names = ['layer2.1.conv2', 'layer3.1.conv2', 'layer4.0.conv2']  # Early, middle, and late layers
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types are 'mlp', 'cnn', and 'resnet18'.")
    
    return layer_names

def run_analysis_phase(
    search_string: Optional[str] = None,
    results_dir: Optional[Path] = None,
    dataset_type: Optional[str] = None
) -> Path:
    """Run analysis phase with direct consumption of evaluation results.
    
    Args:
        search_string: Optional search string to find results directory
        results_dir: Optional path to results directory
        dataset_type: Optional dataset type to use for configuration
        
    Returns:
        Path to results directory
    """
    config = get_default_config(dataset_type=dataset_type)
    config, results_dir = get_config_and_results_dir(base_dir=Path(config['base_dir']), results_dir=results_dir, search_string=search_string)
    
    # Set up logger
    logger = setup_logger(results_dir)
    
    logger.info("\n=== Starting Analysis Phase ===")
    logger.info(f"Using results directory: {results_dir}")
    
    # Load evaluation results directly (using new format)
    results_file = results_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    generate_plots(results, plots_dir, results_dir)
    
    logger.info(f"\n=== Analysis Phase Completed ===")
    logger.info(f"Results saved to: {plots_dir}")
    
    return results_dir

def run_model_class_experiment(
    model_types: List[str] = ['mlp', 'cnn'],
    class_counts: List[int] = [2, 3, 5, 10],
    dataset_types: List[str] = ['mnist'],
    attack_types: List[str] = ['fgsm', 'pgd'],
    base_config: Optional[Dict[str, Any]] = None,
    testing_mode: bool = False 
) -> Dict[str, Dict[int, Path]]:
    """Run training phase across model types and class counts.
    
    Args:
        model_types: List of model types to test ('mlp', 'cnn')
        class_counts: List of class counts to test
        dataset_types: List of dataset types to test
        attack_types: List of attack types to test ('fgsm', 'pgd')
        base_config: Base configuration (uses default if None)
        testing_mode: Whether to run in testing mode
    Returns:
        Dictionary mapping model types to dictionaries mapping class counts to result directories
    """
    # Store result directories
    results = {model_type: {} for model_type in model_types}
    
    # Run experiments for each dataset, model type and class count
    for dataset_type in dataset_types:
        # Get base configuration if not provided
        if base_config is None:
            base_config = get_default_config(testing_mode=testing_mode, dataset_type=dataset_type)
        
        for model_type in model_types:
            for n_classes in class_counts:
                for attack_type in attack_types:
                    print(f"\n{'='*50}")
                    print(f"Running experiment: {model_type.upper()} with {n_classes} classes on {dataset_type} using {attack_type}")
                    print(f"{'='*50}\n")
                    
                    # Select classes based on class count
                    selected_classes = tuple(range(n_classes))
                    
                    # Update configuration for this experiment
                    experiment_config = update_config(base_config, {
                        'model': {'model_type': model_type},
                        'dataset': {'selected_classes': selected_classes},
                        'adversarial': {'attack_type': attack_type},
                        'training': {'n_epochs': 1 if testing_mode else 100}
                    })

                    results_dir = run_training_phase(experiment_config)
                    try: 
                        run_evaluation_phase(results_dir=results_dir, dataset_type=dataset_type)
                        run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)
                    except Exception as e:
                        print(f"Error running all phases for {model_type} with {n_classes} classes using {attack_type}: {e}")
                    
                    # Store result directory
                    if n_classes not in results[model_type]:
                        results[model_type][n_classes] = {}
                    results[model_type][n_classes][attack_type] = results_dir
    
    return results


 
def quick_test(model_type: str = 'cnn', dataset_type: str = "mnist", testing_mode: bool = True):
    # Quick test configuration
    config = get_default_config(dataset_type=dataset_type, testing_mode=testing_mode, selected_classes=(0, 1, 2))
    
    # Run a single training experiment with minimal settings
    print("Running quick test with minimal settings...")
    
    # Customize config for quick test using update_config function
    config = update_config(config, {
        'model': {
            'model_type': model_type
        },
        'adversarial': {
            'attack_type': 'fgsm',
            'train_epsilons': [0.05],  # Fewer epsilon values
            'test_epsilons': [0.0, 0.05],  # Fewer test epsilon values
            'n_runs': 1  # Single run
        }
    })

    print(config)

    # Run training phase
    results_dir = run_training_phase(config)
    
    # Run evaluation on the trained model
    run_evaluation_phase(results_dir=results_dir, dataset_type=dataset_type)
    
    # Generate analysis plots
    run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)
    
    print(f"Quick test completed. Results in: {results_dir}")

# %%
if __name__ == "__main__":

    # experiments to start:
    # - epsilon scaling and limits:
    # mnist cnn 2-class fgsm - 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 - 100 epochs, 1 run  




    # Run quick test
    # quick_test(model_type='mlp', dataset_type="mnist", testing_mode=True)
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




    # Run single experiment: capacity scaling
    # dataset_type = "mnist"
    # for hidden_dim in [16, 32, 64, 128, 256]:
    #     comment = f"h{hidden_dim}"
    #     config = get_default_config(testing_mode=False, dataset_type=dataset_type)
    #     config = update_config(config, {
    #         'model': {
    #             'model_type': 'mlp',
    #             'hidden_dim': hidden_dim 
    #         },
    #         'dataset': {
    #             'selected_classes': tuple(i for i in range(10)) 
    #         },
    #         'training': {
    #             'n_epochs': 100 
    #         },
    #         'adversarial': {
    #             'train_epsilons': [0.0, 0.1, 0.2],
    #             'test_epsilons': [0.0, 0.1, 0.2],
    #             'n_runs': 2,
    #             'attack_type': 'fgsm',
    #         },
    #         'comment': comment
    #     })
    #     results_dir = run_training_phase(config)
    #     run_evaluation_phase(results_dir=results_dir, dataset_type=dataset_type)
    #     run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)

    # run_evaluation_phase(results_dir=results_dir, dataset_type='mnist')
    # run_analysis_phase(results_dir=results_dir, dataset_type='mnist')
    # run_evaluation_phase(search_string="mlp_2-class", dataset_type="cifar10")
    # run_analysis_phase(search_string="cnn_2-class")
    
    # Run model comparison experiment
        # model_types = ['cnn']
    # class_counts = [10]
    # datasets = ['mnist']
    # attack_types = ['fgsm']
    # run_model_class_experiment(model_types, class_counts, datasets, attack_types, testing_mode=False)
    model_types = ['cnn', 'mlp']
    class_counts = [2, 3, 5, 10]
    datasets = ['mnist']
    attack_types = ['fgsm', 'pgd']
    run_model_class_experiment(model_types, class_counts, datasets, attack_types, testing_mode=False)
    
    model_types = ['resnet18']
    class_counts = [2, 3, 5, 10]
    datasets = ['cifar10']
    attack_types = ['fgsm', 'pgd']
    run_model_class_experiment(model_types, class_counts, datasets, attack_types, testing_mode=False)


    # model_types = ['resnet18']
    # class_counts = [2, 10]
    # datasets = ['cifar10']
    # attack_types = ['fgsm', 'pgd']
    # run_model_class_experiment(model_types, class_counts, datasets, attack_types, testing_mode=False)

    # model_types = ['mlp']
    # class_counts = [3, 5, 10]
    # datasets = ['mnist']
    # run_model_class_experiment(model_types, class_counts, datasets, testing_mode=False)

    # model_types = ['resnet18', 'cnn']
    # class_counts = [2, 3, 5, 10]
    # datasets = ['cifar10']
    # run_model_class_experiment(model_types, class_counts, datasets, testing_mode=False)


    # Run single experiment
    # config = get_default_config(testing_mode=False)
    # config['model']['model_type'] = 'mlp'
    # config['model']['hidden_dim'] = 32
    # config['dataset']['selected_classes'] = (0, 1)
    # config['training']['n_epochs'] = 100
    # config['adversarial']['train_epsilons'] = [0.0, 0.1, 0.2]
    # config['adversarial']['n_runs'] = 1
    # results_dir = run_training_phase(config)
    # run_evaluation_phase(results_dir=results_dir)
    # run_analysis_phase(results_dir=results_dir)
# %%
