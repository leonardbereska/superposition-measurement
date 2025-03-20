#%%
"""Main experiment runner for adversarial robustness vs superposition."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

import os
# change working directory to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

from config import (
    MODEL_CONFIG, DATASET_CONFIG, TRAINING_CONFIG, 
    ADVERSARIAL_CONFIG, RESULTS_DIR, DEVICE, DATA_DIR, SAE_CONFIG
)
from datasets import MNISTDataset
from metrics import measure_superposition
from training import SuperpositionExperiment, evaluate_adversarial_robustness
from utils import (
    set_seed, plot_robustness_curve, plot_feature_counts, 
    plot_feature_vs_robustness, save_results_to_json, json_serializer,
)

def run_experiment(
    epsilon: float,
    run_idx: int,
    save_dir: Union[str, Path],
    model_type: str = 'mlp',
    hidden_dim: Optional[int] = None,
    image_size: Optional[int] = None,
    binary_digits: Optional[tuple] = None,
    n_epochs: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Run a single experiment with specified parameters.
    
    Args:
        epsilon: Adversarial training strength
        run_idx: Run index for reproducibility
        save_dir: Directory to save results
        model_type: Model type ('mlp' or 'cnn')
        hidden_dim: Hidden dimension (defaults to config)
        image_size: Image size (defaults to config)
        binary_digits: Binary digits (defaults to config)
        n_epochs: Number of epochs (defaults to config)
        seed: Random seed
        
    Returns:
        Dictionary with experiment results
    """
    # Set defaults from config
    hidden_dim = hidden_dim or MODEL_CONFIG['hidden_dim']
    image_size = image_size or DATASET_CONFIG['target_size']
    binary_digits = binary_digits or DATASET_CONFIG['binary_digits']
    n_epochs = n_epochs or TRAINING_CONFIG['n_epochs']
    
    # Set seed for reproducibility
    if seed is not None:
        set_seed(seed + run_idx)
    
    # Create datasets
    train_dataset = MNISTDataset(
        target_size=image_size,
        train=True,
        binary_digits=binary_digits,
        root=DATA_DIR
    )
    test_dataset = MNISTDataset(
        target_size=image_size,
        train=False,
        binary_digits=binary_digits,
        root=DATA_DIR
    )
    
    # Create experiment directory
    save_dir = Path(save_dir) / f"eps_{epsilon}/run_{run_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    exp = SuperpositionExperiment(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        model_type=model_type,
        hidden_dim=hidden_dim,
        image_size=image_size,
        device=DEVICE
    )
    
    # Train model
    exp.train(
        n_epochs=n_epochs,
        epsilon=epsilon,
        measure_frequency=5,
        measure_superposition=False
    )
    
    # Save experiment
    exp.save(save_dir / "experiment.pt")
    
    # Measure superposition
    layer_name = 'relu' if model_type.lower() == 'mlp' else 'features.0'
    metrics = measure_superposition(
        exp.model, 
        exp.train_loader, 
        layer_name,
        expansion_factor=SAE_CONFIG['expansion_factor'],
        l1_lambda=SAE_CONFIG['l1_lambda'],
        batch_size=SAE_CONFIG['batch_size'],
        learning_rate=SAE_CONFIG['learning_rate'],
        n_epochs=SAE_CONFIG['n_epochs'],
        save_dir=save_dir
    )
    
    # Measure robustness
    test_epsilons = ADVERSARIAL_CONFIG['test_epsilons']
    robustness = evaluate_adversarial_robustness(
        exp.model, 
        exp.val_loader, 
        test_epsilons
    )
    
    # Calculate average robustness
    avg_robustness = sum(robustness.values()) / len(robustness)
    
    # Return results
    results = {
        'feature_count': metrics['feature_count'],
        **{f'accuracy_for_{test_eps}': robustness[test_eps] for test_eps in test_epsilons},
        'robustness_score': avg_robustness
    }
    
    # Save results
    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4, default=json_serializer)
    
    return results

def run_superposition_study(
    train_epsilons: Optional[List[float]] = None,
    test_epsilons: Optional[List[float]] = None,
    n_runs: Optional[int] = None,
    model_type: str = MODEL_CONFIG['model_type'],
    save_dir: Optional[Union[str, Path]] = None,
    plot_results: bool = True
) -> Dict[str, Dict[float, List[float]]]:
    """Run experiments across different adversarial training strengths.
    
    Args:
        train_epsilons: List of training epsilons
        test_epsilons: List of testing epsilons
        n_runs: Number of runs per epsilon
        model_type: Model type ('mlp' or 'cnn')
        save_dir: Directory to save results
        plot_results: Whether to plot results
        
    Returns:
        Dictionary with results
    """
    # Set defaults from config
    train_epsilons = train_epsilons or ADVERSARIAL_CONFIG['train_epsilons']
    test_epsilons = test_epsilons or ADVERSARIAL_CONFIG['test_epsilons']
    n_runs = n_runs or ADVERSARIAL_CONFIG['n_runs']
    
    # Create results directory
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = Path(RESULTS_DIR) / f"{model_type}_adversarial_training_{timestamp}"
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results
    results = {
        'feature_count': {},
        'robustness_score': {},
    }
    
    for eps in test_epsilons:
        results[f'accuracy_for_{eps}'] = {}
    
    # Run experiments
    for eps in train_epsilons:
        print(f"Running experiments for epsilon={eps}")
        
        for metric in results:
            results[metric][eps] = []
        
        for run in range(n_runs):
            print(f"Run {run+1}/{n_runs}")
            
            # Run experiment
            run_results = run_experiment(
                epsilon=eps,
                run_idx=run,
                save_dir=save_dir,
                model_type=model_type
            )
            
            # Store results
            for metric in results:
                results[metric][eps].append(run_results[metric])
            
            print(f"Feature Count: {run_results['feature_count']:.2f}")
            print(f"Robustness Score: {run_results['robustness_score']:.2f}")
            print("-" * 40)
    
    # Save results to JSON
    save_results_to_json(results, save_dir / "results.json")
   
    # Save metadata
    metadata = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model_config": MODEL_CONFIG,
        "dataset_config": DATASET_CONFIG,
        "training_config": TRAINING_CONFIG,
        "adversarial_config": ADVERSARIAL_CONFIG,
        "sae_config": SAE_CONFIG,
    }
    
    with open(save_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4, default=json_serializer)
    
    # Plot results
    if plot_results:
        plot_feature_counts(
            results['feature_count'],
            title="Feature Count vs Adversarial Training Strength",
            save_path=save_dir / "feature_count_vs_training.pdf"
        )
        
        plot_robustness_curve(
            {eps: results['robustness_score'][eps] for eps in results['robustness_score']},
            title="Model Robustness vs Training Strength",
            save_path=save_dir / "robustness_vs_training.pdf"
        )
        
        plot_feature_vs_robustness(
            results['feature_count'],
            results['robustness_score'],
            title="Feature Count vs Model Robustness",
            save_path=save_dir / "feature_count_vs_robustness.pdf"
        )
    
    return results

if __name__ == "__main__":
    # Run the full study
    results = run_superposition_study()
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 40)
    print("Training ε | Feature Count | Robustness Score")
    print("-" * 40)
    
    for eps in sorted(results['feature_count'].keys()):
        feature_mean = np.mean(results['feature_count'][eps])
        feature_std = np.std(results['feature_count'][eps])
        robust_mean = np.mean(results['robustness_score'][eps])
        robust_std = np.std(results['robustness_score'][eps])
        
        print(f"{eps:.2f} | {feature_mean:.2f}±{feature_std:.2f} | {robust_mean:.2f}±{robust_std:.2f}")
# %%
