# %%
"""Main script for running adversarial robustness experiments with a functional approach."""

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

# Import configuration
from config import get_default_config, update_config

# Import components for different phases
from datasets import create_dataloaders
from training import train_model, evaluate_model_performance
from evaluation import evaluate_feature_organization, load_model, json_serializer
from analysis import generate_plots, create_summary, create_report

def setup_results_dir(config: Dict[str, Any]) -> Path:
    """Set up results directory with timestamp.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Path to results directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Determine class information for directory name
    dataset_config = config['dataset']
    model_type = config['model']['model_type']
    
    if dataset_config['selected_classes'] is not None:
        class_info = "".join(map(str, dataset_config['selected_classes']))
        n_classes = len(dataset_config['selected_classes'])
    else:
        class_info = "all"
        n_classes = 10  # Default for MNIST/CIFAR
    
    # Create directory path
    results_dir = Path(config['results_dir']) / f"{timestamp}_{model_type}_{n_classes}-class"
    results_dir.mkdir(parents=True, exist_ok=True)
    
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

def find_latest_results_dir(config: Dict[str, Any]) -> Optional[Path]:
    """Find most recent results directory.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Path to latest results directory or None if not found
    """
    model_type = config['model']['model_type']
    results_dirs = sorted(Path(config['results_dir']).glob(f"*_{model_type}_*"), key=os.path.getmtime)
    
    if not results_dirs:
        print("No results directories found.")
        return None
    
    return results_dirs[-1]

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
            model, history = train_model(
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
                'history': history,
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
            
            print(f"Run completed with validation accuracy: {history['val_accuracy'][-1]:.2f}%")
    
    print(f"\n=== Training Phase Completed ===")
    print(f"Results saved to: {results_dir}")
    
    return results_dir

def run_evaluation_phase(
    config: Dict[str, Any], 
    results_dir: Optional[Path] = None,
    measure_mixed: bool = False
) -> Path:
    """Run evaluation phase.
    
    Args:
        config: Experiment configuration
        results_dir: Optional path to results directory
        measure_mixed: Whether to use mixed clean/adversarial distribution
        
    Returns:
        Path to results directory
    """
    print("\n=== Starting Evaluation Phase ===\n")
    
    # Determine results directory if not provided
    if results_dir is None:
        results_dir = find_latest_results_dir(config)
        if results_dir is None:
            raise ValueError("No results directory found. Please run training phase first.")
    
    print(f"Using results directory: {results_dir}")
    
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
    feature_results = {}
    robustness_results = {}
    
    # Evaluate each epsilon
    for epsilon in config['adversarial']['train_epsilons']:
        print(f"\nEvaluating models trained with epsilon={epsilon}")
        
        eps_dir = results_dir / f"eps_{epsilon}"
        feature_results[epsilon] = []
        robustness_results[epsilon] = []
        
        # Evaluate each run
        for run_idx in range(config['adversarial']['n_runs']):
            run_dir = eps_dir / f"run_{run_idx}"
            eval_dir = run_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)
            
            # try:
            # Load model
            model = load_model(
                model_path=run_dir / "model.pt",
                device=config['device']
            )
            
            # Measure feature organization
            if measure_mixed:
                feature_metrics = evaluate_feature_organization(
                    model=model,
                    train_loader=train_loader,
                    layer_name=layer_name,
                    epsilon=epsilon,
                    use_mixed_distribution=True,
                    expansion_factor=config['sae']['expansion_factor'],
                    l1_lambda=config['sae']['l1_lambda'],
                    batch_size=config['sae']['batch_size'],
                    learning_rate=config['sae']['learning_rate'],
                    n_epochs=config['sae']['n_epochs'],
                    save_dir=eval_dir
                )
                # Use the feature count from mixed SAE on mixed distribution
                feature_count = feature_metrics['mixed']['mixed']['feature_count']
            else:
                feature_metrics = evaluate_feature_organization(
                    model=model,
                    train_loader=train_loader,
                    layer_name=layer_name,
                    expansion_factor=config['sae']['expansion_factor'],
                    l1_lambda=config['sae']['l1_lambda'],
                    batch_size=config['sae']['batch_size'],
                    learning_rate=config['sae']['learning_rate'],
                    n_epochs=config['sae']['n_epochs'],
                    save_dir=eval_dir
                )
                feature_count = feature_metrics['feature_count']
            
            # Load robustness results
            with open(run_dir / "robustness.json", 'r') as f:
                robustness = json.load(f)
            
            # Calculate average robustness
            avg_robustness = sum(float(v) for v in robustness.values()) / len(robustness)
            
            # Store results
            feature_results[epsilon].append(feature_count)
            robustness_results[epsilon].append(avg_robustness)
            
            # Save evaluation results
            with open(eval_dir / "results.json", 'w') as f:
                json.dump({
                    'feature_count': feature_count,
                    'robustness_score': avg_robustness,
                    **{f'accuracy_for_{eps}': robustness[str(eps)] for eps in config['adversarial']['test_epsilons']}
                }, f, indent=4, default=json_serializer)
            
            print(f"Completed evaluation for run {run_idx+1}")
            print(f"Feature Count: {feature_count:.2f}")
            print(f"Average Robustness: {avg_robustness:.2f}%")
            
            # except Exception as e:
            #     print(f"Error evaluating run {run_idx}: {e}")
    
    # Save combined results
    with open(results_dir / "evaluation_results.json", 'w') as f:
        json.dump({
            'feature_count': {str(eps): values for eps, values in feature_results.items()},
            'robustness_score': {str(eps): values for eps, values in robustness_results.items()}
        }, f, indent=4, default=json_serializer)
    
    print(f"\n=== Evaluation Phase Completed ===")
    print(f"Results saved to: {results_dir}")
    
    return results_dir

def run_analysis_phase(
    config: Dict[str, Any], 
    results_dir: Optional[Path] = None,
    create_html_report: bool = False
) -> Path:
    """Run analysis phase.
    
    Args:
        config: Experiment configuration
        results_dir: Optional path to results directory
        create_html_report: Whether to create HTML report
        
    Returns:
        Path to results directory
    """
    print("\n=== Starting Analysis Phase ===\n")
    
    # Determine results directory if not provided
    if results_dir is None:
        results_dir = find_latest_results_dir(config)
        if results_dir is None:
            raise ValueError("No results directory found. Please run training phase first.")
    
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
    generate_plots(processed_results, plots_dir)
    
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

def run_all_phases(
    config: Dict[str, Any] = None,
    measure_mixed: bool = False,
    create_html_report: bool = True
) -> Path:
    """Run all experiment phases sequentially.
    
    Args:
        config: Optional custom configuration
        measure_mixed: Whether to use mixed distribution in evaluation
        create_html_report: Whether to create HTML report
        
    Returns:
        Path to results directory
    """
    # Use default config if not provided
    if config is None:
        config = get_default_config()
    
    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Run training phase
    results_dir = run_training_phase(config)
    
    # Run evaluation phase
    run_evaluation_phase(config, results_dir, measure_mixed)
    
    # Run analysis phase
    run_analysis_phase(config, results_dir, create_html_report)
    
    return results_dir

# %%
# Example usage
if __name__ == "__main__":
    # Get configuration with testing mode for quick runs
    config = get_default_config(testing_mode=True)
    
    # Run specific phase (uncomment as needed)
    # results_dir = run_training_phase(config)
    # results_dir = run_evaluation_phase(config, measure_mixed=False)
    # results_dir = run_evaluation_phase(config, measure_mixed=True)
    results_dir = run_analysis_phase(config, create_html_report=True)
    
    # Or run all phases
    # results_dir = run_all_phases(config, measure_mixed=False, create_html_report=True)
    
    print(f"Experiment completed. Results in: {results_dir}")
# %%

    