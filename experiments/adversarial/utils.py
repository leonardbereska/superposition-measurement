"""Utility functions for adversarial robustness experiments."""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union

def json_serializer(obj):
    """Serialize objects for JSON serialization."""
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int64) or isinstance(obj, np.int32):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Add more type conversions as needed
    return str(obj)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def get_results_dir(experiment_name: str) -> Path:
    """Get path to results directory."""
    results_dir = Path("../../results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def visualize_adversarial_example(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    idx: int,
    epsilon: float = 0.1,
    targeted: bool = False,
    target_label: Optional[torch.Tensor] = None,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Visualize original image, perturbation, and adversarial example.
    
    Args:
        model: Trained model
        dataset: Dataset containing images
        idx: Index of image to visualize
        epsilon: Perturbation size
        targeted: If True, optimize toward target_label
        target_label: Label to target if targeted=True
        save_path: Path to save figure
    """
    from training import generate_adversarial_example
    
    image, label = dataset[idx]
    device = next(model.parameters()).device
    
    # Generate adversarial example
    perturbed_image, pred, perturbation = generate_adversarial_example(
        model, image, label, epsilon, targeted, target_label
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    fontsize = 14
    
    # Original image
    axes[0].imshow(image.detach().cpu(), cmap='gray')
    axes[0].set_title(f'Original\nLabel: {label.item()}', fontsize=fontsize)
    axes[0].axis('off')
    
    # Perturbation
    perturbation_plot = axes[1].imshow(
        perturbation.detach().cpu(), cmap='RdBu', vmin=-epsilon, vmax=epsilon
    )
    axes[1].set_title(f'Perturbation\nε = {epsilon}', fontsize=fontsize)
    axes[1].axis('off')
    
    # Adversarial example
    axes[2].imshow(perturbed_image.detach().cpu(), cmap='gray')
    axes[2].set_title(f'Adversarial\nPred: {int(pred.item())}', fontsize=fontsize)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.colorbar(perturbation_plot, ax=axes[1])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

def plot_robustness_curve(
    results: Dict[float, List[float]],
    title: str = 'Adversarial Robustness',
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot robustness curve across different epsilon values.
    
    Args:
        results: Dictionary mapping epsilon values to list of accuracies
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    # Colors
    colors = ['#BA898A', '#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']
    fontsize = 14
    
    # Calculate mean and std for each epsilon
    epsilons = sorted(results.keys())
    means = [np.mean(results[eps]) for eps in epsilons]
    stds = [np.std(results[eps]) for eps in epsilons]
    
    # Plot
    plt.errorbar(
        epsilons, means, yerr=stds, fmt='o-',
        color=colors[3], linewidth=2, capsize=5, markersize=8
    )
    
    plt.xlabel('Perturbation Size (ε)', fontsize=fontsize)
    plt.ylabel('Accuracy (%)', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.grid(True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

def plot_feature_counts(
    results: Dict[float, List[float]],
    title: str = 'Feature Count vs Adversarial Training Strength',
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot feature counts across different epsilon values.
    
    Args:
        results: Dictionary mapping epsilon values to list of feature counts
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    # Colors
    colors = ['#BA898A', '#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']
    fontsize = 14
    
    # Calculate mean and std for each epsilon
    epsilons = sorted(results.keys())
    means = [np.mean(results[eps]) for eps in epsilons]
    stds = [np.std(results[eps]) for eps in epsilons]
    
    # Plot
    plt.errorbar(
        epsilons, means, yerr=stds, fmt='o-',
        color=colors[3], linewidth=2, capsize=5, markersize=8
    )
    
    plt.xlabel('Adversarial Training Strength (ε)', fontsize=fontsize)
    plt.ylabel('Effective Feature Count', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.grid(True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

def plot_feature_vs_robustness(
    feature_results: Dict[float, List[float]],
    robustness_results: Dict[float, List[float]],
    title: str = 'Feature Count vs Model Robustness',
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Plot feature counts vs robustness.
    
    Args:
        feature_results: Dictionary mapping epsilon values to list of feature counts
        robustness_results: Dictionary mapping epsilon values to list of accuracies
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    # Colors
    colors = ['#BA898A', '#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']
    fontsize = 14
    
    # Calculate mean and std for each epsilon
    epsilons = sorted(feature_results.keys())
    feature_means = [np.mean(feature_results[eps]) for eps in epsilons]
    feature_stds = [np.std(feature_results[eps]) for eps in epsilons]
    robustness_means = [np.mean(robustness_results[eps]) for eps in epsilons]
    robustness_stds = [np.std(robustness_results[eps]) for eps in epsilons]
    
    # Plot
    plt.errorbar(
        feature_means, robustness_means, 
        xerr=feature_stds, yerr=robustness_stds,
        fmt='.', capsize=5, color=colors[3], markersize=12
    )
    
    # Add epsilon annotations
    for i, eps in enumerate(epsilons):
        plt.annotate(
            f'ε={eps}', 
            (feature_means[i], robustness_means[i]),
            xytext=(8, -20), 
            textcoords='offset points', 
            color=colors[3], 
            fontsize=fontsize
        )
    
    plt.xlabel('Effective Feature Count', fontsize=fontsize)
    plt.ylabel('Average Robustness', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.grid(True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()

def save_results_to_json(
    results: Dict[str, Dict[float, List[float]]],
    save_path: Union[str, Path]
) -> None:
    """Save results to JSON file.
    
    Args:
        results: Dictionary mapping metric names to dictionaries mapping 
                epsilon values to lists of results
        save_path: Path to save JSON file
    """
    
    # Create raw data structure
    raw_data = []
    
    for eps in sorted(results['feature_count'].keys()):
        for i in range(len(results['feature_count'][eps])):
            row = {'epsilon': eps}
            for metric in results:
                row[f'{metric}'] = results[metric][eps][i]
                
            raw_data.append(row)
    
    # Calculate mean and std for each epsilon and metric
    summary = []
    for eps in sorted(results['feature_count'].keys()):
        row = {'epsilon': eps}
        for metric in results:
            values = results[metric][eps]
            row[f'{metric}_mean'] = np.mean(values)
            row[f'{metric}_std'] = np.std(values)
        
        summary.append(row)
    
    # Save to JSON
    with open(str(save_path), 'w') as f:
        json.dump(summary, f, indent=4, default=json_serializer)