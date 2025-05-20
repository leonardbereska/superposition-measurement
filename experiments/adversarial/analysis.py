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

from config import get_default_config
from utils import load_config



class ScientificPlotStyle:
    """Standard style settings for scientific visualizations in our papers."""
    
    # Color palette - soft muted colors for data series
    COLORS = ['#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']  # yellow, green, blue, purple
    ERROR_COLOR = '#BA898A'  # soft red for error indicators
    REFERENCE_LINE_COLOR = '#8B5C5D'  # dark red for reference lines
    
    # Typography - large sizes for readability
    FONT_SIZE_TITLE = 48     # plot titles
    FONT_SIZE_LABELS = 36    # axis labels
    FONT_SIZE_TICKS = 36     # tick labels
    FONT_SIZE_LEGEND = 28    # legend text
    
    # Plot elements
    MARKER_SIZE = 15         # data point size
    LINE_WIDTH = 5.0         # line thickness
    CAPSIZE = 12             # error bar cap size
    CAPTHICK = 5.0           # error bar cap thickness
    GRID_ALPHA = 0.3         # grid transparency
    
    # Figure dimensions
    FIGURE_SIZE = (12, 10)   # standard figure size
    COMBINED_FIG_SIZE = (20, 10)  # two-panel figure size
    
    @staticmethod
    def apply_axis_style(ax, title, xlabel, ylabel, legend=True):
        """Apply consistent styling to a matplotlib axis."""
        ax.set_title(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
        ax.set_ylabel(ylabel, fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
        ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
        ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
        if legend:
            ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND, loc='best')
        return ax
    
    @staticmethod
    def errorbar_kwargs(color_idx=0):
        """Return standard error bar parameters."""
        return {
            'marker': 'o',
            'color': ScientificPlotStyle.COLORS[color_idx % len(ScientificPlotStyle.COLORS)],
            'markersize': ScientificPlotStyle.MARKER_SIZE,
            'linewidth': ScientificPlotStyle.LINE_WIDTH,
            'capsize': ScientificPlotStyle.CAPSIZE,
            'capthick': ScientificPlotStyle.CAPTHICK,
            'elinewidth': ScientificPlotStyle.LINE_WIDTH
        }

def plot_robustness_curve(
    results: Dict[float, List[float]],
    title: str = 'Adversarial Robustness',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
    color_idx: int = 0
) -> plt.Figure:
    """Plot robustness curve across different epsilon values.
    
    Args:
        results: Dictionary mapping epsilon values to list of accuracies
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        color_idx: Index for color selection from ScientificPlotStyle
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std for each epsilon
    epsilons = sorted(results.keys())
    means = [np.mean(results[eps]) for eps in epsilons]
    stds = [np.std(results[eps]) for eps in epsilons]
    
    # Get style parameters
    style_kwargs = ScientificPlotStyle.errorbar_kwargs(color_idx)
    
    # Plot
    ax.errorbar(
        epsilons, means, yerr=stds,
        **style_kwargs
    )
    
    # Apply scientific plot styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel='Perturbation Size (ε)',
        ylabel='Accuracy (%)',
        legend=False
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_combined_feature_counts(
    results: Dict[str, Dict[float, List[float]]],
    title: str = 'Feature Counts vs Adversarial Training Strength',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE
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
    for i, metric_name in enumerate(['clean_feature_count', 'adversarial_feature_count', 'mixed_feature_count']):
        if metric_name not in results:
            continue
            
        metric_data = results[metric_name]
        
        # Calculate mean and std for each epsilon
        epsilons = sorted(metric_data.keys())
        means = [np.mean(metric_data[eps]) for eps in epsilons]
        stds = [np.std(metric_data[eps]) for eps in epsilons]
        
        # Plot with error bars using ScientificPlotStyle
        ax.errorbar(
            epsilons, means, yerr=stds,
            label=labels[metric_name],
            **ScientificPlotStyle.errorbar_kwargs(i)
        )
    
    # Apply scientific plot styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel='Adversarial Training Strength (ε)',
        ylabel='Feature Count'
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_feature_distribution_matrix(
    results: Dict[str, Dict[float, List[float]]],
    epsilon: float,
    title: str = 'Feature Distribution Matrix',
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
    ax.set_ylabel('Feature Count', fontsize=14)
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
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
    color_idx: int = 0,
    add_annotations: bool = True
) -> plt.Figure:
    """Plot feature counts vs robustness.
    
    Args:
        feature_results: Dictionary mapping epsilon values to list of feature counts
        robustness_results: Dictionary mapping epsilon values to list of accuracies
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        color_idx: Index for color selection from ScientificPlotStyle
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
    
    # Get style parameters
    style_kwargs = ScientificPlotStyle.errorbar_kwargs(color_idx)
    
    # Plot
    ax.errorbar(
        feature_means, robustness_means, 
        xerr=feature_stds, yerr=robustness_stds,
        fmt='.', capsize=ScientificPlotStyle.CAPSIZE, 
        markersize=ScientificPlotStyle.MARKER_SIZE,
        color=style_kwargs['color'],
        elinewidth=style_kwargs['linewidth'],
        capthick=ScientificPlotStyle.CAPTHICK
    )
    
    # Add epsilon annotations
    if add_annotations:
        for i, eps in enumerate(epsilons):
            ax.annotate(
                f'ε={eps}', 
                (feature_means[i], robustness_means[i]),
                xytext=(10, -25), 
                textcoords='offset points', 
                color=style_kwargs['color'], 
                fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND
            )
    
    # Apply scientific plot styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel='Feature Count',
        ylabel='Average Robustness',
        legend=False
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def generate_plots(results: Dict, output_dir: Path, results_dir: Path) -> Dict[str, plt.Figure]:
    """Generate analysis plots for results.
    
    Args:
        results: Dictionary with results
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    # load metadata
    metadata = load_config(results_dir)

    # get model type and number of classes
    model_type = metadata['model']['model_type']
    n_classes = len(list(metadata['dataset']['selected_classes']))
    experiment_name = f"{model_type.upper()} {n_classes}-class"  

    def save_path(name):
        experiment_string = f"{model_type}_{n_classes}-class"
        return output_dir / f"{experiment_string}_{name}.pdf"
    

    # Robustness curve
    if 'robustness_score' in results:
        name = "robustness"
        fig = plot_robustness_curve(
            results['robustness_score'],
            title=f"Model Robustness {experiment_name}",
            save_path=save_path(name)
        )
        figures[name] = fig
    
    # Feature vs robustness
    name = "feature_vs_robustness"
    if 'feature_count' in results and 'robustness_score' in results:
        fig = plot_feature_vs_robustness(
            results['feature_count'],
            results['robustness_score'],
            title=f"Feature Count vs Model Robustness {experiment_name}",
            save_path=save_path(name)
        )
        figures[name] = fig
    
    # Generate combined feature count plot
    name = "feature_counts"
    fig = plot_combined_feature_counts(
        results,
        title=f"{experiment_name}",
        save_path=save_path(name)
    )
    figures[name] = fig
    
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