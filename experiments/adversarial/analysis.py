# %%
"""Analysis and visualization utilities for adversarial robustness experiments."""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
from datetime import datetime

# %%
def load_results(results_dir: Path) -> Dict:
    """Load experiment results from a directory.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary with loaded results
    """
    results_path = results_dir / "evaluation_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Convert string keys back to floats where needed
        processed_results = {}
        for metric, values in results.items():
            processed_results[metric] = {float(eps): vals for eps, vals in values.items()}
        
        return processed_results
    else:
        # Try to aggregate results from individual runs
        results = {
            'feature_count': {},
            'robustness_score': {},
        }
        
        # Look for epsilon directories
        for eps_dir in results_dir.glob("eps_*"):
            epsilon = float(eps_dir.name.split("_")[1])
            
            # Initialize lists for this epsilon
            for metric in results:
                results[metric][epsilon] = []
            
            # Look for run directories
            for run_dir in eps_dir.glob("run_*"):
                # Look for evaluation results
                eval_path = run_dir / "evaluation/results.json"
                if eval_path.exists():
                    with open(eval_path, 'r') as f:
                        run_results = json.load(f)
                    
                    # Add to results
                    for metric in results:
                        if metric in run_results:
                            results[metric][epsilon].append(run_results[metric])
        
        return results

# %%
def load_metadata(results_dir: Path) -> Dict:
    """Load experiment metadata.
    
    Args:
        results_dir: Directory containing experiment metadata
        
    Returns:
        Dictionary with metadata
    """
    metadata_path = results_dir / "config.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        return {}

# %%
def calculate_statistics(results: Dict) -> Dict:
    """Calculate statistics for results.
    
    Args:
        results: Dictionary with results
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    for metric, values in results.items():
        stats[metric] = {}
        
        for epsilon, trials in values.items():
            if not trials:
                continue
                
            stats[metric][epsilon] = {
                'mean': np.mean(trials),
                'std': np.std(trials),
                'min': np.min(trials),
                'max': np.max(trials),
                'n': len(trials)
            }
    
    return stats

# %%
def create_summary(results: Dict) -> pd.DataFrame:
    """Create a summary dataframe of results.
    
    Args:
        results: Dictionary with results
        
    Returns:
        Pandas DataFrame with summarized results
    """
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Prepare data for DataFrame
    data = []
    
    # Get all epsilons across all metrics
    all_epsilons = set()
    for metric in stats:
        all_epsilons.update(stats[metric].keys())
    
    # Sort epsilons
    all_epsilons = sorted(all_epsilons)
    
    # Create rows
    for epsilon in all_epsilons:
        row = {'epsilon': epsilon}
        
        for metric in stats:
            if epsilon in stats[metric]:
                row[f'{metric}_mean'] = stats[metric][epsilon]['mean']
                row[f'{metric}_std'] = stats[metric][epsilon]['std']
        
        data.append(row)
    
    # Create DataFrame
    return pd.DataFrame(data)

# %%
def plot_robustness_curve(
    results: Dict[float, List[float]],
    title: str = 'Adversarial Robustness',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    color: str = '#88A7B2'
) -> plt.Figure:
    """Plot robustness curve across different epsilon values.
    
    Args:
        results: Dictionary mapping epsilon values to list of accuracies
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        color: Line color
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std for each epsilon
    epsilons = sorted(results.keys())
    means = [np.mean(results[eps]) for eps in epsilons]
    stds = [np.std(results[eps]) for eps in epsilons]
    
    # Plot
    ax.errorbar(
        epsilons, means, yerr=stds, fmt='o-',
        color=color, linewidth=2, capsize=5, markersize=8
    )
    
    ax.set_xlabel('Perturbation Size (ε)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Set tick parameters
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_combined_feature_counts(
    results: Dict[str, Dict[float, List[float]]],
    title: str = 'Feature Counts vs Adversarial Training Strength',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot all feature counts (clean, adversarial, mixed) on one plot.
    
    Args:
        results: Dictionary with feature count results
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors for different distributions
    colors = {
        'clean_feature_count': '#F0BE5E',         # Yellow
        'adversarial_feature_count': '#94B9A3',   # Green
        'mixed_feature_count': '#88A7B2',         # Blue
    }
    
    # Labels for the legend
    labels = {
        'clean_feature_count': 'Clean Inputs',
        'adversarial_feature_count': 'Adversarial Inputs',
        'mixed_feature_count': 'Mixed Inputs'
    }
    
    # Get set of all epsilons across all metrics
    all_epsilons = set()
    for metric in results:
        if metric.endswith('_feature_count'):
            all_epsilons.update(results[metric].keys())
    
    # Sort epsilons
    all_epsilons = sorted(all_epsilons)
    
    # Plot each feature count metric
    for metric_name in ['clean_feature_count', 'adversarial_feature_count', 'mixed_feature_count']:
        if metric_name not in results:
            continue
            
        metric_data = results[metric_name]
        
        # Calculate mean and std for each epsilon
        epsilons = sorted(metric_data.keys())
        means = [np.mean(metric_data[eps]) for eps in epsilons]
        stds = [np.std(metric_data[eps]) for eps in epsilons]
        
        # Plot with error bars
        ax.errorbar(
            epsilons, means, yerr=stds, fmt='o-',
            color=colors[metric_name], linewidth=2, capsize=5, markersize=8,
            label=labels[metric_name]
        )
    
    # Customize plot
    ax.set_xlabel('Adversarial Training Strength (ε)', fontsize=14)
    ax.set_ylabel('Effective Feature Count', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Set tick parameters
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_feature_distribution_matrix(
    results: Dict[str, Dict[float, List[float]]],
    epsilon: float,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot a heatmap of feature counts for a specific epsilon value.
    
    Args:
        results: Dictionary with feature count results
        epsilon: Epsilon value to visualize
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Get feature counts for this epsilon
    labels = ['Clean', 'Adversarial', 'Mixed']
    feature_counts = []
    
    for metric in ['clean_feature_count', 'adversarial_feature_count', 'mixed_feature_count']:
        if metric in results and epsilon in results[metric]:
            mean_count = np.mean(results[metric][epsilon])
            feature_counts.append(mean_count)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create data matrix (just a 1D array in this case)
    data = np.array(feature_counts).reshape(-1, 1)
    
    # Create heatmap
    im = ax.imshow(data, cmap='YlGnBu')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Feature Count', rotation=-90, va="bottom", fontsize=12)
    
    # Show all ticks and label them
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xticks([])  # No x-ticks needed
    
    # Add feature count text in each cell
    for i in range(len(labels)):
        ax.text(0, i, f"{feature_counts[i]:.2f}", 
                ha="center", va="center", color="black", fontsize=14)
    
    # Set title and adjust layout
    ax.set_title(f'Feature Distribution (ε={epsilon})', fontsize=16)
    fig.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig
# %%
def plot_feature_counts(
    results: Dict[float, List[float]],
    title: str = 'Feature Count vs Adversarial Training Strength',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    color: str = '#88A7B2'
) -> plt.Figure:
    """Plot feature counts across different epsilon values.
    
    Args:
        results: Dictionary mapping epsilon values to list of feature counts
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        color: Line color
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std for each epsilon
    epsilons = sorted(results.keys())
    means = [np.mean(results[eps]) for eps in epsilons]
    stds = [np.std(results[eps]) for eps in epsilons]
    
    # Plot
    ax.errorbar(
        epsilons, means, yerr=stds, fmt='o-',
        color=color, linewidth=2, capsize=5, markersize=8
    )
    
    ax.set_xlabel('Adversarial Training Strength (ε)', fontsize=14)
    ax.set_ylabel('Effective Feature Count', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Set tick parameters
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

# %%
def plot_feature_vs_robustness(
    feature_results: Dict[float, List[float]],
    robustness_results: Dict[float, List[float]],
    title: str = 'Feature Count vs Model Robustness',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    color: str = '#88A7B2',
    add_annotations: bool = True
) -> plt.Figure:
    """Plot feature counts vs robustness.
    
    Args:
        feature_results: Dictionary mapping epsilon values to list of feature counts
        robustness_results: Dictionary mapping epsilon values to list of accuracies
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        color: Point color
        add_annotations: Whether to add epsilon annotations
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std for each epsilon
    epsilons = sorted(set(feature_results.keys()) & set(robustness_results.keys()))
    feature_means = [np.mean(feature_results[eps]) for eps in epsilons]
    feature_stds = [np.std(feature_results[eps]) for eps in epsilons]
    robustness_means = [np.mean(robustness_results[eps]) for eps in epsilons]
    robustness_stds = [np.std(robustness_results[eps]) for eps in epsilons]
    
    # Plot
    ax.errorbar(
        feature_means, robustness_means, 
        xerr=feature_stds, yerr=robustness_stds,
        fmt='.', capsize=5, color=color, markersize=12
    )
    
    # Add epsilon annotations
    if add_annotations:
        for i, eps in enumerate(epsilons):
            ax.annotate(
                f'ε={eps}', 
                (feature_means[i], robustness_means[i]),
                xytext=(8, -20), 
                textcoords='offset points', 
                color=color, 
                fontsize=12
            )
    
    ax.set_xlabel('Effective Feature Count', fontsize=14)
    ax.set_ylabel('Average Robustness', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Set tick parameters
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

# %%
def generate_plots(results: Dict, output_dir: Path) -> Dict[str, plt.Figure]:
    """Generate analysis plots for results.
    
    Args:
        results: Dictionary with results
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    
    # Feature count plot
    if 'feature_count' in results:
        fig = plot_feature_counts(
            results['feature_count'],
            title="Feature Count vs Adversarial Training Strength",
            save_path=output_dir / "feature_count.pdf"
        )
        figures['feature_count'] = fig
    
    # Robustness curve
    if 'robustness_score' in results:
        fig = plot_robustness_curve(
            results['robustness_score'],
            title="Model Robustness vs Training Strength",
            save_path=output_dir / "robustness.pdf"
        )
        figures['robustness'] = fig
    
    # Feature vs robustness
    if 'feature_count' in results and 'robustness_score' in results:
        fig = plot_feature_vs_robustness(
            results['feature_count'],
            results['robustness_score'],
            title="Feature Count vs Model Robustness",
            save_path=output_dir / "feature_vs_robustness.pdf"
        )
        figures['feature_vs_robustness'] = fig
    
    # Generate individual feature count plots
    for metric in results:
        if metric.endswith('_feature_count'):
            metric_name = metric.replace('_feature_count', '')
            fig = plot_feature_counts(
                results[metric],
                title=f"{metric_name.capitalize()} Feature Count vs Adversarial Training",
                save_path=output_dir / f"{metric_name}_feature_count.pdf"
            )
            figures[f"{metric_name}_feature_count"] = fig
    
    # Generate combined feature count plot
    fig = plot_combined_feature_counts(
        results,
        title="Feature Counts Across Input Distributions",
        save_path=output_dir / "combined_feature_counts.pdf"
    )
    figures['combined_feature_counts'] = fig
    
    # Generate feature distribution matrix
    fig = plot_feature_distribution_matrix(
        results,
        title="Feature Distribution Matrix",
        save_path=output_dir / "feature_distribution_matrix.pdf"
    )
    figures['feature_distribution_matrix'] = fig
    
    # Individual test epsilon plots
    for key in results:
        if key.startswith('accuracy_for_'):
            eps = key.split('_')[-1]
            fig = plot_robustness_curve(
                results[key],
                title=f"Accuracy for ε={eps} vs Training Strength",
                save_path=output_dir / f"accuracy_for_{eps}.pdf"
            )
            figures[key] = fig
    
    return figures

# %%
def create_report(
    results: Dict,
    results_dir: Path,
    plots_dir: Path
) -> Path:
    """Create a comprehensive HTML report.
    
    Args:
        results: Dictionary with results
        results_dir: Directory to save report
        plots_dir: Directory containing plots
        
    Returns:
        Path to the report
    """
    # Load metadata
    metadata = load_metadata(results_dir)
    
    # Create summary
    summary_df = create_summary(results)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Superposition Experiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metadata {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Superposition Experiment Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        
        <h2>Experiment Metadata</h2>
        <div class="metadata">
            <pre>{json.dumps(metadata, indent=4)}</pre>
        </div>
        
        <h2>Summary Statistics</h2>
        {summary_df.to_html(index=False)}
        
        <h2>Visualizations</h2>
        
        <div class="plot">
            <h3>Feature Count vs Adversarial Training Strength</h3>
            <img src="plots/feature_count.pdf" alt="Feature Count Plot">
        </div>
        
        <div class="plot">
            <h3>Model Robustness vs Training Strength</h3>
            <img src="plots/robustness.pdf" alt="Robustness Plot">
        </div>
        
        <div class="plot">
            <h3>Feature Count vs Model Robustness</h3>
            <img src="plots/feature_vs_robustness.pdf" alt="Feature vs Robustness Plot">
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    report_path = results_dir / "report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

# %%
def compare_results(
    results_dirs: Dict[str, Path],
    output_dir: Path,
    plot_type: str = 'feature_count'
) -> Dict[str, plt.Figure]:
    """Compare results across different experiments.
    
    Args:
        results_dirs: Dictionary mapping experiment names to result directories
        output_dir: Directory to save comparison plots
        plot_type: Type of plot to generate ('feature_count', 'robustness', 'feature_vs_robustness')
        
    Returns:
        Dictionary mapping plot names to figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    
    # Load results
    all_results = {}
    for name, path in results_dirs.items():
        all_results[name] = load_results(path)
    
    # Colors for different experiments
    colors = ['#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']
    
    if plot_type == 'feature_count':
        # Create feature count comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (name, results) in enumerate(all_results.items()):
            if 'feature_count' not in results:
                continue
                
            feature_data = results['feature_count']
            
            epsilons = sorted(feature_data.keys())
            means = [np.mean(feature_data[eps]) for eps in epsilons]
            stds = [np.std(feature_data[eps]) for eps in epsilons]
            
            ax.errorbar(
                epsilons, means, yerr=stds, fmt='o-',
                color=colors[i % len(colors)], linewidth=2, capsize=5, markersize=8,
                label=name
            )
        
        ax.set_xlabel('Adversarial Training Strength (ε)', fontsize=14)
        ax.set_ylabel('Effective Feature Count', fontsize=14)
        ax.set_title('Feature Count Comparison', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set tick parameters
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / "feature_count_comparison.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        
        figures['feature_count'] = fig
    
    elif plot_type == 'robustness':
        # Create robustness comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (name, results) in enumerate(all_results.items()):
            if 'robustness_score' not in results:
                continue
                
            robust_data = results['robustness_score']
            
            epsilons = sorted(robust_data.keys())
            means = [np.mean(robust_data[eps]) for eps in epsilons]
            stds = [np.std(robust_data[eps]) for eps in epsilons]
            
            ax.errorbar(
                epsilons, means, yerr=stds, fmt='o-',
                color=colors[i % len(colors)], linewidth=2, capsize=5, markersize=8,
                label=name
            )
        
        ax.set_xlabel('Adversarial Training Strength (ε)', fontsize=14)
        ax.set_ylabel('Average Robustness (%)', fontsize=14)
        ax.set_title('Robustness Comparison', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set tick parameters
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / "robustness_comparison.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        
        figures['robustness'] = fig
    
    elif plot_type == 'feature_vs_robustness':
        # Create feature vs robustness comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (name, results) in enumerate(all_results.items()):
            if 'feature_count' not in results or 'robustness_score' not in results:
                continue
                
            feature_data = results['feature_count']
            robust_data = results['robustness_score']
            
            # Find common epsilons
            epsilons = sorted(set(feature_data.keys()) & set(robust_data.keys()))
            
            feature_means = [np.mean(feature_data[eps]) for eps in epsilons]
            robust_means = [np.mean(robust_data[eps]) for eps in epsilons]
            
            ax.plot(
                feature_means, robust_means, 'o-',
                color=colors[i % len(colors)], linewidth=2, markersize=8,
                label=name
            )
            
            # Add epsilon annotations
            for j, eps in enumerate(epsilons):
                ax.annotate(
                    f'ε={eps}', 
                    (feature_means[j], robust_means[j]),
                    xytext=(5, 5), 
                    textcoords='offset points', 
                    color=colors[i % len(colors)], 
                    fontsize=10
                )
        
        ax.set_xlabel('Effective Feature Count', fontsize=14)
        ax.set_ylabel('Average Robustness (%)', fontsize=14)
        ax.set_title('Feature Count vs Robustness', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set tick parameters
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        # Save figure
        save_path = output_dir / "feature_vs_robustness_comparison.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        
        figures['feature_vs_robustness'] = fig
    
    return figures