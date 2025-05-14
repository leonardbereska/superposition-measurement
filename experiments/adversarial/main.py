# %%
"""Main script for running adversarial robustness experiments."""

import os
import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import configuration, datasets, utils
from config import get_default_config, update_config
from datasets_adversarial import create_dataloaders
from utils import json_serializer, setup_results_dir, find_results_dir, save_config, get_config_and_results_dir

# Import components for different phases 
from training import train_model, evaluate_model_performance
from evaluation import load_model, measure_superposition_on_mixed_distribution
from analysis import generate_plots, create_summary, create_report

def run_training_phase(config: Dict[str, Any], results_dir: Optional[Path] = None) -> Path:
    """Run training phase.
    
    Args:
        config: Experiment configuration
        results_dir: Optional custom results directory
        
    Returns:
        Path to results directory
    """
    print("\n=== Starting Training Phase ===\n")
    
    # Create results directory if not provided
    if results_dir is None:
        results_dir = setup_results_dir(config)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Save configuration
    save_config(config, results_dir)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_type=config['dataset']['dataset_type'],
        batch_size=config['training']['batch_size'],
        selected_classes=config['dataset']['selected_classes'],
        target_size=config['dataset']['target_size'],
        data_dir=config['data_dir']
    )
    
    # Train models with different adversarial strengths
    for epsilon in config['adversarial']['train_epsilons']:
        print(f"\nTraining with epsilon={epsilon}")
        
        # Create directory for this epsilon
        eps_dir = results_dir / f"eps_{epsilon}"
        eps_dir.mkdir(exist_ok=True)
        
        # Run multiple training runs
        for run_idx in range(config['adversarial']['n_runs']):
            print(f"Run {run_idx+1}/{config['adversarial']['n_runs']}")
            
            # Set random seed for reproducibility
            run_seed = 42 + run_idx
            torch.manual_seed(run_seed)
            
            # Create directory for this run
            run_dir = eps_dir / f"run_{run_idx}"
            run_dir.mkdir(exist_ok=True)
            
            # Train model
            model = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model_type=config['model']['model_type'],
                hidden_dim=config['model']['hidden_dim'],
                image_size=config['model']['image_size'],
                n_epochs=config['training']['n_epochs'],
                learning_rate=config['training']['learning_rate'],
                epsilon=epsilon,
                device=config['device']
            )
            
            # Save model and history
            torch.save({
                'model_state': model.state_dict(),
                'config': {
                    'model_type': config['model']['model_type'],
                    'hidden_dim': config['model']['hidden_dim'],
                    'image_size': config['model']['image_size'],
                    'epsilon': epsilon,
                    'seed': run_seed
                }
            }, run_dir / "model.pt")
            
            # Evaluate on test epsilons
            robustness_results = evaluate_model_performance(
                model=model,
                dataloader=val_loader,
                epsilons=config['adversarial']['test_epsilons'],
                device=config['device']
            )
            
            # Save results
            with open(run_dir / "robustness.json", 'w') as f:
                json.dump(robustness_results, f, indent=4, default=json_serializer)
            
    print(f"\n=== Training Phase Completed ===")
    print(f"Results saved to: {results_dir}")
    
    return results_dir

def run_evaluation_phase(
    search_string: Optional[Path] = None,
    results_dir: Optional[Path] = None
) -> Path:
    """Run evaluation phase.
    
    Args:
        search_string: search string to find results directory
        
    Returns:
        Path to results directory
    """
    print("\n=== Starting Evaluation Phase ===\n")
    config = get_default_config()
    config, results_dir = get_config_and_results_dir(base_dir=Path(config['base_dir']), results_dir=results_dir, search_string=search_string)
    
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_type=config['dataset']['dataset_type'],
        batch_size=config['training']['batch_size'],
        selected_classes=config['dataset']['selected_classes'],
        target_size=config['dataset']['target_size'],
        data_dir=config['data_dir']
    )
    
    # Determine layer name based on model type
    layer_name = 'relu' if config['model']['model_type'].lower() == 'mlp' else 'features.1'
    
    # Store overall results
    clean_feature_results = {}
    adversarial_feature_results = {}
    mixed_feature_results = {}
    robustness_results = {}
    
    # Evaluate each epsilon
    for epsilon in config['adversarial']['train_epsilons']:
        print(f"\nEvaluating models trained with epsilon={epsilon}")
        
        eps_dir = results_dir / f"eps_{epsilon}"
        clean_feature_results[epsilon] = []
        adversarial_feature_results[epsilon] = []
        mixed_feature_results[epsilon] = []
        robustness_results[epsilon] = []
        
        # Evaluate each run
        for run_idx in range(config['adversarial']['n_runs']):
            run_dir = eps_dir / f"run_{run_idx}"
            eval_dir = run_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)
            
            # Load model
            model = load_model(
                model_path=run_dir / "model.pt",
                device=config['device']
            )
            
            # Measure feature organization
            epsilon = float(epsilon)
            feature_metrics = measure_superposition_on_mixed_distribution(
                model=model,
                dataloader=train_loader,
                layer_name=layer_name,
                epsilon=epsilon,
                expansion_factor=config['sae']['expansion_factor'],
                l1_lambda=config['sae']['l1_lambda'],
                batch_size=config['sae']['batch_size'],
                learning_rate=config['sae']['learning_rate'],
                n_epochs=config['sae']['n_epochs'],
                save_dir=eval_dir
            )
            
            # Extract feature counts
            clean_count = feature_metrics['clean_feature_count']
            adv_count = feature_metrics['adversarial_feature_count']
            mixed_count = feature_metrics['mixed_feature_count']
            
            # Load robustness results
            with open(run_dir / "robustness.json", 'r') as f:
                robustness = json.load(f)
            
            # Calculate average robustness
            avg_robustness = sum(float(v) for v in robustness.values()) / len(robustness)
            
            # Store results
            clean_feature_results[epsilon].append(clean_count)
            adversarial_feature_results[epsilon].append(adv_count)
            mixed_feature_results[epsilon].append(mixed_count)
            robustness_results[epsilon].append(avg_robustness)
            
            # Save evaluation results
            with open(eval_dir / "results.json", 'w') as f:
                json.dump({
                    'clean_feature_count': clean_count,
                    'adversarial_feature_count': adv_count,
                    'mixed_feature_count': mixed_count,
                    'robustness_score': avg_robustness,
                    **{f'accuracy_for_{eps}': robustness[str(eps)] for eps in config['adversarial']['test_epsilons']}
                }, f, indent=4, default=json_serializer)
            
            print(f"Completed evaluation for run {run_idx+1}")
            print(f"Clean Feature Count: {clean_count:.2f}")
            print(f"Adversarial Feature Count: {adv_count:.2f}")
            print(f"Mixed Feature Count: {mixed_count:.2f}")
            print(f"Average Robustness: {avg_robustness:.2f}%")
    
    # Save combined results
    with open(results_dir / "evaluation_results.json", 'w') as f:
        json.dump({
            'clean_feature_count': {str(eps): values for eps, values in clean_feature_results.items()},
            'adversarial_feature_count': {str(eps): values for eps, values in adversarial_feature_results.items()},
            'mixed_feature_count': {str(eps): values for eps, values in mixed_feature_results.items()},
            'robustness_score': {str(eps): values for eps, values in robustness_results.items()}
        }, f, indent=4, default=json_serializer)
    
    print(f"\n=== Evaluation Phase Completed ===")
    print(f"Results saved to: {results_dir}")
    
    return results_dir

def run_analysis_phase(
    search_string: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    create_html_report: bool = False
) -> Path:
    """Run analysis phase.
    
    Args:
        search_string: Optional search string to find results directory
        results_dir: Optional path to results directory
        create_html_report: Whether to create HTML report
        
    Returns:
        Path to results directory
    """
    print("\n=== Starting Analysis Phase ===\n")

    config = get_default_config()
    config, results_dir = get_config_and_results_dir(base_dir=Path(config['base_dir']), results_dir=results_dir, search_string=search_string)
    print(f"Using results directory: {results_dir}")
    
    # Load evaluation results
    try:
        with open(results_dir / "evaluation_results.json", 'r') as f:
            results = json.load(f)
        
        # Convert string keys back to floats
        processed_results = {}
        for metric, values in results.items():
            processed_results[metric] = {float(eps): vals for eps, vals in values.items()}
    except FileNotFoundError:
        print("Evaluation results not found. Aggregating from individual runs...")
        
        # Aggregate results from individual runs
        feature_results = {}
        robustness_results = {}
        
        for eps_dir in results_dir.glob("eps_*"):
            epsilon = float(eps_dir.name.split("_")[1])
            
            feature_results[epsilon] = []
            robustness_results[epsilon] = []
            
            for run_dir in eps_dir.glob("run_*"):
                eval_results_path = run_dir / "evaluation/results.json"
                if eval_results_path.exists():
                    with open(eval_results_path, 'r') as f:
                        run_results = json.load(f)
                    
                    feature_results[epsilon].append(run_results['feature_count'])
                    robustness_results[epsilon].append(run_results['robustness_score'])
        
        processed_results = {
            'feature_count': feature_results,
            'robustness_score': robustness_results
        }
    
    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    generate_plots(processed_results, plots_dir, results_dir)
    
    # Create summary
    print("Creating summary...")
    summary_df = create_summary(processed_results)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    
    # Create report if requested
    if create_html_report:
        print("Creating HTML report...")
        report_path = create_report(processed_results, results_dir, plots_dir)
        print(f"Report created at: {report_path}")
    
    print(f"\n=== Analysis Phase Completed ===")
    print(f"Results saved to: {plots_dir}")
    
    return results_dir

def run_model_class_experiment(
    model_types: List[str] = ['mlp', 'cnn'],
    class_counts: List[int] = [2, 3, 5, 10],
    base_config: Optional[Dict[str, Any]] = None,
    testing_mode: bool = False 
) -> Dict[str, Dict[int, Path]]:
    """Run training phase across model types and class counts.
    
    Args:
        model_types: List of model types to test ('mlp', 'cnn')
        class_counts: List of class counts to test
        base_config: Base configuration (uses default if None)
        testing_mode: Whether to run in testing mode
    Returns:
        Dictionary mapping model types to dictionaries mapping class counts to result directories
    """
    # Get base configuration if not provided
    if base_config is None:
        base_config = get_default_config(testing_mode=testing_mode)
    
    # Store result directories
    results = {model_type: {} for model_type in model_types}
    
    # Run experiments for each model type and class count
    for model_type in model_types:
        for n_classes in class_counts:
            print(f"\n{'='*50}")
            print(f"Running experiment: {model_type.upper()} with {n_classes} classes")
            print(f"{'='*50}\n")
            
            # Select classes based on class count
            selected_classes = tuple(range(n_classes))
            
            # Update configuration for this experiment
            experiment_config = update_config(base_config, {
                'model': {'model_type': model_type},
                'dataset': {'selected_classes': selected_classes}
            })

            results_dir = run_training_phase(experiment_config)
            try: 
                run_evaluation_phase(results_dir=results_dir)
                run_analysis_phase(results_dir=results_dir)
            except Exception as e:
                print(f"Error running all phases for {model_type} with {n_classes} classes: {e}")
            
            # Store result directory
            results[model_type][n_classes] = results_dir
    
    return results

def evaluate_all_experiments(config: Optional[Dict[str, Any]] = None, base_dir: Optional[Path] = None, measure_mixed: bool = True):
    """Evaluate all experiment models in the results directory.
    
    Args:
        config: Experiment configuration
        base_dir: Base directory containing experiment folders
        measure_mixed: Whether to use mixed clean/adversarial distribution
    """
    # Get default config
    if config is None:
        config = get_default_config()
    
    if base_dir is None: 
        base_dir = Path(config['base_dir'])
    
    print(f"\n=== Evaluating All Experiments in {base_dir} ===\n")
    
    # Find all experiment directories
    experiment_dirs = [d for d in base_dir.glob("*") if d.is_dir()]
    
    for exp_dir in experiment_dirs:
        try:
            print(f"\nEvaluating experiment: {exp_dir.name}")
            
            # Load the experiment's config.json 
            config_path = exp_dir / "config.json"
            
            with open(config_path, 'r') as f:
                exp_config = json.load(f)
            # Update the default config with the experiment's config
            config = update_config(config, exp_config)
            
            # Run evaluation on this experiment directory
            run_evaluation_phase(
                config=config,
                search_string=exp_dir,
                measure_mixed=measure_mixed
            )
            
            print(f"Completed evaluation for: {exp_dir.name}")
        except Exception as e:
            print(f"Error evaluating {exp_dir.name}: {str(e)}")
    
    print("\n=== All Evaluations Complete ===")
 
def quick_test(model_type: str = 'cnn'):
    # Quick test configuration
    config = get_default_config(testing_mode=True)
    
    # Run a single training experiment with minimal settings
    print("Running quick test with minimal settings...")
    
    # Customize config for quick test
    config['model']['model_type'] = model_type
    config['model']['hidden_dim'] = 16  # Smaller model
    config['training']['n_epochs'] = 2  # Fewer epochs
    config['adversarial']['train_epsilons'] = [0.0, 0.1]  # Fewer epsilon values
    config['adversarial']['test_epsilons'] = [0.0, 0.1]  # Fewer test epsilon values
    config['adversarial']['n_runs'] = 1  # Single run
    
    # Run training phase
    results_dir = run_training_phase(config)
    
    # Run evaluation on the trained model
    run_evaluation_phase(results_dir=results_dir)
    
    # Generate analysis plots
    run_analysis_phase(results_dir=results_dir)
    
    print(f"Quick test completed. Results in: {results_dir}")

# %%
if __name__ == "__main__":
    # Run quick test
    # quick_test(model_type='cnn')
    # run_evaluation_phase(search_string="cnn_2-class")
    # run_analysis_phase(search_string="cnn_2-class")
    
    # Run model comparison experiment
    model_types = ['cnn', 'mlp']
    class_counts = [2, 3, 5, 10]
    run_model_class_experiment(model_types, class_counts, testing_mode=False)

    # Run single experiment
    # config = get_default_config(testing_mode=False)
    # config['model']['model_type'] = 'mlp'
    # config['dataset']['selected_classes'] = (0, 1)
    # config['model']['hidden_dim'] = 32
    # config['training']['n_epochs'] = 100
    # config['adversarial']['train_epsilons'] = [0.0, 0.1, 0.2]
    # config['adversarial']['n_runs'] = 1
    # results_dir = run_training_phase(config)
    # run_evaluation_phase(results_dir=results_dir)
    # run_analysis_phase(results_dir=results_dir)
# %%
