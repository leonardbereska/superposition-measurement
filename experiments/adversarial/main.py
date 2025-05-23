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
from training import train_model, evaluate_model_performance, load_model, save_model, TrainingConfig
from evaluation import measure_superposition
from analysis import generate_plots
from attacks import AttackConfig, generate_adversarial_examples, get_adversarial_dataloader
from models import ModelConfig, create_model
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
        use_early_stopping=config['training'].get('use_early_stopping', True)
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
    log(f"\tModel structure: {config['model']}")
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
    if model_type == 'mlp':  # for MNIST
        layer_names = [
            'relu1', # post-ReLU after each fc
            'relu2', 
            'relu3',
        ]
    elif model_type == 'cnn': # for MNIST
        layer_names = [
            'conv1.2', # post-ReLU after first conv (28×28)
            'conv2.2', # post-ReLU after second conv (14×14)
            'fc1.1', # post-ReLU after first fc 
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
            'layer1.1.bn2',   # low_level: basic patterns (32×32) (pre-ReLU, residual)
            'layer1.1',       # low_level: basic patterns (32×32) (post-ReLU, combines skip and residual)
            'layer2.1.bn2',   # mid_level: object parts (16×16) (pre-ReLU, residual)
            'layer2.1',       # mid_level: object parts (16×16) (post-ReLU, combines skip and residual)
            'layer3.1.bn2',   # high_level: object concepts (8×8) (pre-ReLU, residual)
            'layer3.1',       # high_level: object concepts (8×8) (post-ReLU, combines skip and residual)
            'layer4.1.bn2',   # semantic: class-relevant features (4×4) (pre-ReLU, residual)
            'layer4.1',       # semantic: class-relevant features (4×4) (post-ReLU, combines skip and residual)
            'avgpool'         # Global features (512ch, 1×1)
        ]
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
    model_types: List[str],
    class_counts: List[int],
    dataset_types: List[str],
    attack_types: List[str],
    testing_mode: bool = False
):
    """Run experiments across model types, class counts, and attack types.
    
    Args:
        model_types: List of model types to test
        class_counts: List of class counts to test
        dataset_types: List of dataset types to test
        attack_types: List of attack types to test
        testing_mode: Whether to run in testing mode
    """
    for dataset_type in dataset_types:
        
        for attack_type in attack_types:
            for model_type in model_types:
                for n_classes in class_counts:
                    config = get_default_config(testing_mode=testing_mode, dataset_type=dataset_type, selected_classes=tuple(range(n_classes)))
                    text = f"Running experiment: {model_type.upper()} with {n_classes} classes on {dataset_type.upper()} using {attack_type.upper()}"
                    print(f"\n{'='*len(text)}")
                    print(text)
                    print(f"{'='*len(text)}\n")

                    # if CNN, set hidden_dim to 16
                    # if MLP, set hidden_dim to 32
                    if model_type == 'cnn':
                        config = update_config(config, {
                            'model': {'hidden_dim': 16}
                        })
                    elif model_type == 'mlp':
                        config = update_config(config, {
                            'model': {'hidden_dim': 32}
                        })
                    
                    config = update_config(config, {
                        'model': {'model_type': model_type},
                        'adversarial': {'attack_type': attack_type},
                        'dataset': {'selected_classes': tuple(range(n_classes))},
                    })

                    results_dir = run_training_phase(config)
                    run_evaluation_phase(results_dir=results_dir, dataset_type=dataset_type)
                    run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)

 
def quick_test(model_type: str = None, dataset_type: str = "cifar10", testing_mode: bool = True):
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
    config = update_config(config, {
        'comment': f'quick_test_{dataset_type}_{config["model"]["model_type"]}'
    })

    # Print key configuration
    print(f"Model: {config['model']['model_type']}")
    print(f"Dataset: {config['dataset']['dataset_type']}")
    print(f"Attack: {config['adversarial']['attack_type']} with {config['adversarial']['pgd_steps']} steps")
    print(f"Epsilon: {config['adversarial']['train_epsilons']}")
    print(f"Epochs: {config['training']['n_epochs']}")

    # Run training phase
    results_dir = run_training_phase(config)
    
    # Run evaluation on the trained model
    run_evaluation_phase(results_dir=results_dir, dataset_type=dataset_type)
    
    # Generate analysis plots
    run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)
    
    print(f"Quick test completed. Results in: {results_dir}")
    return results_dir

# %%
if __name__ == "__main__":
    
    # run_model_class_experiment(
    #     model_types=['cnn', 'mlp'],
    #     class_counts=[2, 3, 5, 10],
    #     dataset_types=['mnist'],
    #     attack_types=['pgd'],
    #     testing_mode=False
    # )
    
    # MNIST experiment
    # config = get_default_config(
    #     dataset_type="mnist", 
    #     testing_mode=False,
    #     selected_classes=(0, 1)
    # )
    # config = update_config(config, {
    #     'comment': 'standard_pgd40_eps0.3'
    # })
    # results_dir = run_training_phase(config)
    # run_evaluation_phase(results_dir=results_dir, dataset_type=dataset_type)
    # run_analysis_phase(results_dir=results_dir, dataset_type=dataset_type)

    # === PREVIOUS EXPERIMENTS (commented out) ===
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

    # NOTE: to run
    # 2025-05-23 10-55
    # base_channels = [16, 32, 64, 128]
    # for base_channel in base_channels:
    #     config = get_default_config(testing_mode=False, dataset_type="mnist")
    #     config = update_config(config, {
    #         'model': {'model_type': 'cnn', 'hidden_dim': base_channel}
    #     })
    #     results_dir = run_training_phase(config)
    #     run_evaluation_phase(results_dir=results_dir, dataset_type="mnist")
    #     run_analysis_phase(results_dir=results_dir, dataset_type="mnist")


    # 2025-05-23 16-05  NOTE current experiment
    # model_types = ['mlp', 'cnn']
    # class_counts = [2, 3, 5, 10]
    # dataset_types = ['mnist']
    # attack_types = ['fgsm', 'pgd']
    # run_model_class_experiment(model_types, class_counts, dataset_types, attack_types, testing_mode=False)

    # 2025-05-23 16-04  NOTE current experiment
    # model_types = ['resnet18']
    # class_counts = [2, 3, 5, 10]
    # dataset_types = ['cifar10']
    # attack_types = ['fgsm', 'pgd']
    # run_model_class_experiment(model_types, class_counts, dataset_types, attack_types, testing_mode=False)


    # NOTE: previous experiments
    # 2025-05-22 17-26 
    # model_types = ['cnn', 'resnet18']
    # class_counts = [2, 3, 5, 10]
    # datasets = ['cifar10']
    # attack_types = ['pgd']
    # run_model_class_experiment(model_types, class_counts, datasets, attack_types, testing_mode=False)

    # 2025-05-21 10-25
    # model_types = ['cnn', 'mlp']
    # class_counts = [2, 3, 5, 10]
    # datasets = ['mnist']
    # attack_types = ['fgsm', 'pgd']
    # run_model_class_experiment(model_types, class_counts, datasets, attack_types, testing_mode=False)
    
    # model_types = ['resnet18']
    # class_counts = [2, 3, 5, 10]
    # datasets = ['cifar10']
    # attack_types = ['fgsm', 'pgd']
    # run_model_class_experiment(model_types, class_counts, datasets, attack_types, testing_mode=False)

     # 2025-05-20 10-25
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

# %%