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




def plot_llc_vs_robustness(
    llc_results: Dict[float, List[float]],
    robustness_results: Dict[float, List[float]],
    title: str = 'LLC vs Model Robustness',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
    color_idx: int = 0,
    add_annotations: bool = True
) -> plt.Figure:
    """Plot LLC values vs robustness.
    
    Args:
        llc_results: Dictionary mapping epsilon values to list of LLC values
        robustness_results: Dictionary mapping epsilon values to list of accuracies
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        color_idx: Index for color selection
        add_annotations: Whether to add epsilon annotations
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std for each epsilon
    epsilons = sorted(set(llc_results.keys()) & set(robustness_results.keys()))
    llc_means = [np.mean(llc_results[eps]) for eps in epsilons]
    llc_stds = [np.std(llc_results[eps]) for eps in epsilons]
    robustness_means = [np.mean(robustness_results[eps]) for eps in epsilons]
    robustness_stds = [np.std(robustness_results[eps]) for eps in epsilons]
    
    # Get style parameters
    style_kwargs = ScientificPlotStyle.errorbar_kwargs(color_idx)
    
    # Plot
    ax.errorbar(
        llc_means, robustness_means, 
        xerr=llc_stds, yerr=robustness_stds,
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
                (llc_means[i], robustness_means[i]),
                xytext=(10, -25), 
                textcoords='offset points', 
                color=style_kwargs['color'], 
                fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND
            )
    
    # Apply scientific plot styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel='LLC Value',
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
        results_dir: Path to the results directory for loading metadata
        
    Returns:
        Dictionary mapping plot names to figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    # Load metadata
    metadata = load_config(results_dir)

    # Get model type and number of classes
    model_type = metadata['model']['model_type']
    n_classes = len(list(metadata['dataset']['selected_classes']))
    experiment_name = f"{model_type.upper()} {n_classes}-class"  

    def save_path(name):
        experiment_string = f"{model_type}_{n_classes}-class"
        return output_dir / f"{experiment_string}_{name}.pdf"
    
    # Process results for each layer
    if 'layers' in results[next(iter(results))]:
        # New format with layer-wise analysis
        return generate_plots_from_nested_results(results, output_dir, model_type, n_classes)
    
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

def generate_plots_from_nested_results(results: Dict, output_dir: Path, model_type: str, n_classes: int) -> Dict[str, plt.Figure]:
    """Generate plots from the nested results structure with layer-wise analysis.
    
    Args:
        results: Nested results dictionary
        output_dir: Directory to save plots
        model_type: Type of model used
        n_classes: Number of classes in the dataset
        
    Returns:
        Dictionary mapping plot names to figures
    """
    figures = {}
    experiment_name = f"{model_type.upper()} {n_classes}-class"
    
    def save_path(name):
        experiment_string = f"{model_type}_{n_classes}-class"
        return output_dir / f"{experiment_string}_{name}.pdf"
    
    # Extract epsilons and sort them
    epsilons = sorted([float(eps) for eps in results.keys()])
    
    # 1. Generate comprehensive overview plot with all layers and distributions
    if 'layers' in results[str(epsilons[0])]:
        layers = list(results[str(epsilons[0])]['layers'].keys())
        fig = plot_comprehensive_feature_overview(
            results, layers, epsilons,
            title=f"Feature Count Overview - {experiment_name}",
            save_path=save_path("comprehensive_feature_overview")
        )
        figures["comprehensive_feature_overview"] = fig

    # 2. Generate comprehensive robustness overview plot
    # Extract robustness data for all test epsilons
    fig = plot_comprehensive_robustness_overview(
        results, epsilons,
        title=f"Robustness Overview - {experiment_name}",
        save_path=save_path("comprehensive_robustness_overview")
    )
    figures["comprehensive_robustness_overview"] = fig
    
    return figures

def plot_comprehensive_feature_overview(
    results: Dict, 
    layers: List[str], 
    epsilons: List[float], 
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 12)
) -> plt.Figure:
    """Plot comprehensive overview of feature counts across all layers and distributions.
    
    Args:
        results: Nested results dictionary
        layers: List of layer names
        epsilons: List of epsilon values
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define line styles for different data distributions
    distributions = {
        "clean_feature_count": {"linestyle": "-", "name": "Clean "},
        "adv_feature_count": {"linestyle": "--", "name": "Adversarial"}
    }
    
    # Plot each layer and distribution combination
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    line_count = 0
    
    for i, layer_name in enumerate(layers):
        for j, (dist_key, dist_props) in enumerate(distributions.items()):
            # Extract data for this layer and distribution
            layer_data = {}
            for eps in epsilons:
                layer_data[eps] = results[str(eps)]['layers'][layer_name][dist_key]
            
            # Calculate means and standard deviations
            means = [np.mean(layer_data[eps]) for eps in epsilons]
            stds = [np.std(layer_data[eps]) for eps in epsilons]
            
            # Determine color and marker
            color_idx = i % len(ScientificPlotStyle.COLORS)
            marker = marker_styles[i % len(marker_styles)]
            
            # Create label
            label = f"{layer_name} - {dist_props['name']}"
            
            # Plot
            ax.errorbar(
                epsilons, means, yerr=stds,
                label=label,
                color=ScientificPlotStyle.COLORS[color_idx],
                marker=marker,
                linestyle=dist_props['linestyle'],
                markersize=ScientificPlotStyle.MARKER_SIZE,
                linewidth=ScientificPlotStyle.LINE_WIDTH,
                capsize=ScientificPlotStyle.CAPSIZE,
                capthick=ScientificPlotStyle.CAPTHICK,
                elinewidth=ScientificPlotStyle.LINE_WIDTH
            )
            
            line_count += 1
    
    # Apply styling with adjusted legend
    ax.set_title(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel('Adversarial Training Strength (ε)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.set_ylabel('Feature Count', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    # Adjust legend based on number of entries
    if line_count > 8:
        # Use two-column legend for many entries
        ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND * 0.8, 
                 loc='best', ncol=2)
    else:
        ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_sae_layer_comparison(
    results: Dict, 
    layers: List[str], 
    epsilons: List[float], 
    metric_key: str, 
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE
) -> plt.Figure:
    """Plot feature counts across different layers.
    
    Args:
        results: Nested results dictionary
        layers: List of layer names
        epsilons: List of epsilon values
        metric_key: Which metric to plot ('clean_feature_count' or 'adv_feature_count')
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each layer
    for i, layer_name in enumerate(layers):
        # Extract data for this layer
        layer_data = {}
        for eps in epsilons:
            layer_data[eps] = results[str(eps)]['layers'][layer_name][metric_key]
        
        # Calculate means and standard deviations
        means = [np.mean(layer_data[eps]) for eps in epsilons]
        stds = [np.std(layer_data[eps]) for eps in epsilons]
        
        # Plot
        ax.errorbar(
            epsilons, means, yerr=stds,
            label=layer_name,
            **ScientificPlotStyle.errorbar_kwargs(i % len(ScientificPlotStyle.COLORS))
        )
    
    # Apply styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel='Adversarial Training Strength (ε)',
        ylabel='Feature Count'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_comprehensive_robustness_overview(
    results: Dict, 
    epsilons: List[float], 
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 12)
) -> plt.Figure:
    """Plot comprehensive overview of robustness scores across all test epsilons and training epsilons.
    
    Args:
        results: Nested results dictionary containing detailed_robustness for each training epsilon
        epsilons: List of training epsilon values
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get all test epsilons from the first training epsilon
    test_epsilons = sorted([float(eps) for eps in results[str(epsilons[0])]['detailed_robustness'].keys()])
    
    # Define line styles for different test epsilons
    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    
    for i, test_eps in enumerate(test_epsilons):
        # Extract data for this test epsilon across all training epsilons
        test_data = {}
        for eps in epsilons:
            test_data[eps] = results[str(eps)]['detailed_robustness'][str(test_eps)]
        
        # Calculate means and standard deviations
        means = [np.mean(test_data[eps]) for eps in epsilons]
        stds = [np.std(test_data[eps]) for eps in epsilons]
        
        # Determine color and marker
        color_idx = i % len(ScientificPlotStyle.COLORS)
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]
        
        # Create label
        label = f"Test ε={test_eps}"
        
        # Plot
        ax.errorbar(
            epsilons, means, yerr=stds,
            label=label,
            color=ScientificPlotStyle.COLORS[color_idx],
            marker=marker,
            linestyle=linestyle,
            markersize=ScientificPlotStyle.MARKER_SIZE,
            linewidth=ScientificPlotStyle.LINE_WIDTH,
            capsize=ScientificPlotStyle.CAPSIZE,
            capthick=ScientificPlotStyle.CAPTHICK,
            elinewidth=ScientificPlotStyle.LINE_WIDTH
        )
    
    # Apply styling with adjusted legend
    ax.set_title(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel('Training Epsilon (ε)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.set_ylabel('Robustness Score', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    # Adjust legend based on number of entries
    if len(test_epsilons) > 8:
        # Use two-column legend for many entries
        ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND * 0.8, 
                 loc='best', ncol=2)
    else:
        ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

# ============================================================================
# IMPROVED ARCHITECTURE: SEPARATE SAE AND LLC PLOTTING
# ============================================================================

def generate_sae_plots(results: Dict, output_dir: Path, results_dir: Path) -> Dict[str, plt.Figure]:
    """Generate SAE-specific analysis plots ONLY.
    
    Args:
        results: Dictionary with results
        output_dir: Directory to save plots
        results_dir: Path to the results directory for loading metadata
        
    Returns:
        Dictionary mapping plot names to figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    # Load metadata
    metadata = load_config(results_dir)
    model_type = metadata['model']['model_type']
    n_classes = len(list(metadata['dataset']['selected_classes']))
    experiment_name = f"{model_type.upper()} {n_classes}-class"  

    def save_path(name):
        experiment_string = f"{model_type}_{n_classes}-class"
        return output_dir / f"sae_{experiment_string}_{name}.pdf"
    
    # Extract epsilons and sort them
    epsilons = sorted([float(eps) for eps in results.keys()])
    
    # Check if we have SAE layer data
    if 'layers' in results[str(epsilons[0])]:
        layers = list(results[str(epsilons[0])]['layers'].keys())
        
        # 1. Comprehensive SAE feature overview
        has_sae_data = any('clean_feature_count' in results[str(eps)]['layers'][layer] 
                          for eps in epsilons for layer in layers)
        
        if has_sae_data:
            fig = plot_comprehensive_feature_overview(
                results, layers, epsilons,
                title=f"SAE Feature Count Overview - {experiment_name}",
                save_path=save_path("comprehensive_feature_overview")
            )
            figures["sae_comprehensive_feature_overview"] = fig
            
            # 2. Individual layer comparisons
            for layer_name in layers:
                # Clean vs adversarial for this layer
                clean_data = {}
                adv_data = {}
                
                for eps in epsilons:
                    eps_str = str(eps)
                    if (layer_name in results[eps_str]['layers'] and
                        'clean_feature_count' in results[eps_str]['layers'][layer_name]):
                        clean_data[eps] = results[eps_str]['layers'][layer_name]['clean_feature_count']
                        adv_data[eps] = results[eps_str]['layers'][layer_name]['adv_feature_count']
                
                if clean_data and adv_data:
                    fig = plot_combined_feature_counts(
                        {
                            'clean_feature_count': clean_data,
                            'adversarial_feature_count': adv_data
                        },
                        title=f"SAE Features - {layer_name} - {experiment_name}",
                        save_path=save_path(f"{layer_name.replace('.', '_')}_feature_comparison")
                    )
                    figures[f"sae_{layer_name}_comparison"] = fig
    
    # 3. Robustness overview (this is general, not SAE-specific)
    fig = plot_comprehensive_robustness_overview(
        results, epsilons,
        title=f"Robustness Overview - {experiment_name}",
        save_path=output_dir / f"{model_type}_{n_classes}-class_robustness_overview.pdf"
    )
    figures["robustness_overview"] = fig
    
    return figures


def generate_llc_plots(results: Dict, output_dir: Path, config: Dict, logger) -> Dict[str, plt.Figure]:
    """Generate LLC-specific analysis plots ONLY.
    
    Args:
        results: Dictionary with results
        output_dir: Directory to save plots  
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        Dictionary mapping plot names to figures
    """
    figures = {}
    log = logger.info
    
    # Extract basic info
    epsilons = sorted([float(eps) for eps in results.keys()])
    model_type = config['model']['model_type']
    n_classes = len(config['dataset']['selected_classes'])
    
    def save_path(name):
        experiment_string = f"{model_type}_{n_classes}-class"
        return output_dir / f"llc_{experiment_string}_{name}.png"
    
    # Check if LLC data is available
    has_llc_data = any('checkpoint_llc_analysis' in results[str(eps)] 
                       for eps in epsilons 
                       if 'checkpoint_llc_analysis' in results[str(eps)])
    
    if has_llc_data:
        log("Generating LLC-specific analysis plots...")
        
        # 1. Extract LLC checkpoint evolution data
        llc_checkpoint_evolution = {}
        for eps in epsilons:
            eps_str = str(eps)
            if 'checkpoint_llc_analysis' in results[eps_str]:
                llc_values = []
                for checkpoint_analysis in results[eps_str]['checkpoint_llc_analysis']:
                    if 'epsilon_analysis' in checkpoint_analysis:
                        if eps_str in checkpoint_analysis['epsilon_analysis']:
                            for checkpoint_data in checkpoint_analysis['epsilon_analysis'][eps_str]:
                                llc_values.append(checkpoint_data['llc_mean'])
                
                if llc_values:
                    llc_checkpoint_evolution[eps] = llc_values
        
        # 2. Plot LLC evolution across training for multiple epsilons
        if llc_checkpoint_evolution:
            max_checkpoints = max(len(values) for values in llc_checkpoint_evolution.values())
            checkpoints = list(range(0, max_checkpoints * config['llc']['measure_frequency'], 
                             config['llc']['measure_frequency']))[:max_checkpoints]
            
            fig = plot_llc_across_training_multiple_eps(
                llc_results=llc_checkpoint_evolution,
                checkpoints=checkpoints,
                title="LLC Evolution During Training - Multi-Epsilon Comparison",
                save_path=save_path("evolution_multi_epsilon"),
                n_runs=config['adversarial']['n_runs']
            )
            figures["llc_evolution_multi_epsilon"] = fig
        
        # 3. Clean vs Adversarial LLC comparison  
        clean_llc_data = {}
        adv_llc_data = {}
        
        for eps in epsilons:
            eps_str = str(eps)
            
            # Extract clean LLC data (final values from checkpoints)
            if 'llc_clean' in results[eps_str] and results[eps_str]['llc_clean']:
                clean_llc_data[eps] = [entry['llc_mean'] for entry in results[eps_str]['llc_clean'] 
                                      if entry['llc_mean'] is not None]
            
            # Extract adversarial LLC data (final values from checkpoints)  
            if 'llc_adversarial' in results[eps_str] and results[eps_str]['llc_adversarial']:
                adv_llc_data[eps] = [entry['llc_mean'] for entry in results[eps_str]['llc_adversarial'] 
                                    if entry['llc_mean'] is not None]
        
        if clean_llc_data and adv_llc_data:
            fig = plot_clean_vs_adversarial_llc(
                clean_llc=clean_llc_data,
                adv_llc=adv_llc_data,
                title=f"Clean vs Adversarial LLC - {model_type.upper()} {n_classes}-class",
                save_path=save_path("clean_vs_adversarial")
            )
            figures["llc_clean_vs_adversarial"] = fig
    
    return figures


def generate_combined_sae_llc_plots(results: Dict, output_dir: Path, config: Dict, logger) -> Dict[str, plt.Figure]:
    """Generate combined SAE + LLC analysis plots.
    
    Args:
        results: Dictionary with results
        output_dir: Directory to save plots
        config: Experiment configuration  
        logger: Logger instance
        
    Returns:
        Dictionary mapping plot names to figures
    """
    figures = {}
    log = logger.info
    
    epsilons = sorted([float(eps) for eps in results.keys()])
    model_type = config['model']['model_type']
    n_classes = len(config['dataset']['selected_classes'])
    
    def save_path(name):
        experiment_string = f"{model_type}_{n_classes}-class"
        return output_dir / f"combined_{experiment_string}_{name}.png"
    
    # Check if both SAE and LLC data are available
    has_sae_data = any('layers' in results[str(eps)] and 
                      any('clean_feature_count' in results[str(eps)]['layers'][layer] 
                          for layer in results[str(eps)]['layers'])
                      for eps in epsilons)
    
    has_llc_data = any('checkpoint_llc_analysis' in results[str(eps)] 
                       for eps in epsilons)
    
    if has_sae_data and has_llc_data:
        log("Generating combined SAE + LLC correlation plots...")
        
        # Extract LLC data for correlation
        llc_final_values = {}
        for eps in epsilons:
            eps_str = str(eps)
            if 'checkpoint_llc_analysis' in results[eps_str]:
                # Use final LLC value from checkpoint evolution
                for checkpoint_analysis in results[eps_str]['checkpoint_llc_analysis']:
                    if 'epsilon_analysis' in checkpoint_analysis:
                        if eps_str in checkpoint_analysis['epsilon_analysis']:
                            final_checkpoint = checkpoint_analysis['epsilon_analysis'][eps_str][-1]
                            llc_final_values[eps] = final_checkpoint['llc_mean']
                            break
        
        # Generate correlation plots for each layer
        if 'layers' in results[str(epsilons[0])]:
            layers = list(results[str(epsilons[0])]['layers'].keys())
            
            for layer_name in layers:
                sae_data = {}
                llc_data = {}
                
                for eps in epsilons:
                    eps_str = str(eps)
                    
                    # Get SAE feature counts
                    if (layer_name in results[eps_str]['layers'] and
                        'clean_feature_count' in results[eps_str]['layers'][layer_name]):
                        feature_counts = results[eps_str]['layers'][layer_name]['clean_feature_count']
                        sae_data[eps] = [x for x in feature_counts if x is not None]
                    
                    # Get LLC data
                    if eps in llc_final_values:
                        llc_data[eps] = [llc_final_values[eps]] * len(sae_data.get(eps, [1]))
                
                if sae_data and llc_data:
                    fig = plot_combined_sae_llc_correlation(
                        sae_data=sae_data,
                        llc_data=llc_data,
                        title=f"SAE vs LLC Correlation - {layer_name} - {model_type.upper()} {n_classes}-class",
                        save_path=save_path(f"sae_llc_correlation_{layer_name.replace('.', '_')}")
                    )
                    figures[f"combined_sae_llc_{layer_name}"] = fig
    
    return figures


# ============================================================================
# UPDATED MAIN GENERATE_PLOTS FUNCTION  
# ============================================================================

def generate_plots(results: Dict, output_dir: Path, results_dir: Path) -> Dict[str, plt.Figure]:
    """Generate comprehensive analysis plots with modular architecture.
    
    Args:
        results: Dictionary with results
        output_dir: Directory to save plots
        results_dir: Path to the results directory for loading metadata
        
    Returns:
        Dictionary mapping plot names to figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration for LLC plotting
    config = load_config(results_dir)
    
    # Set up logger (simple print-based for now)
    class SimpleLogger:
        def info(self, msg): print(msg)
    logger = SimpleLogger()
    
    # Generate all plot types
    all_figures = {}
    
    # 1. SAE-specific plots
    sae_figures = generate_sae_plots(results, output_dir, results_dir)
    all_figures.update(sae_figures)
    
    # 2. LLC-specific plots  
    llc_figures = generate_llc_plots(results, output_dir, config, logger)
    all_figures.update(llc_figures)
    
    # 3. Combined SAE + LLC plots
    combined_figures = generate_combined_sae_llc_plots(results, output_dir, config, logger)
    all_figures.update(combined_figures)
    
    return all_figures

# ============================================================================
#  LLC-SPECIFIC PLOTTING FUNCTIONS 
# ============================================================================


def plot_training_evolution(
    training_history: Dict[str, List[float]],
    llc_measurements: List[Dict],
    adversarial_results: Dict[float, float],
    checkpoints: List[int],
    title: str = 'Training Evolution with LLC Analysis',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 20)
) -> plt.Figure:
    """Plot comprehensive training evolution with LLC analysis.
    
    This recreates the original LLCTrackingExperiment.plot_training_evolution() method.
    
    Args:
        training_history: Dictionary with 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
        llc_measurements: List of LLC measurement dictionaries with 'epoch' and 'stats'
        adversarial_results: Dictionary mapping epsilon to accuracy
        checkpoints: List of checkpoint epochs
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
    
    # Get all epochs for metrics measured every epoch
    all_epochs = np.arange(len(training_history['train_loss']))
    
    # First subplot: Accuracy vs LLC
    ax1.plot(all_epochs, training_history['val_accuracy'], 
            label='Validation Accuracy', color='blue', alpha=0.7, linewidth=2)
    ax1.plot(all_epochs, training_history['train_accuracy'], 
            label='Train Accuracy', color='purple', alpha=0.7, linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    
    # Plot LLC measurements with uncertainty bands
    if llc_measurements:
        llc_epochs = [m['epoch'] for m in llc_measurements]
        llc_means = [m['stats']['llc_average_mean'] for m in llc_measurements]
        llc_stds = [m['stats']['llc_average_std'] for m in llc_measurements]
        
        ax1_twin = ax1.twinx()
        # Plot mean line
        ax1_twin.plot(llc_epochs, llc_means, 
                    color='green', label='LLC', marker='o', 
                    linewidth=3, markersize=8)
        # Add uncertainty bands
        ax1_twin.fill_between(llc_epochs,
                            [m - s for m, s in zip(llc_means, llc_stds)],
                            [m + s for m, s in zip(llc_means, llc_stds)],
                            color='green', alpha=0.2)
        ax1_twin.set_ylabel('LLC', color='green', fontsize=14)
        ax1_twin.tick_params(axis='y', labelcolor='green')
    
    # Second subplot: Loss vs LLC
    ax2.plot(all_epochs, training_history['val_loss'], 
            label='Validation Loss', color='orange', alpha=0.7, linewidth=2)
    ax2.plot(all_epochs, training_history['train_loss'], 
            label='Train Loss', color='red', alpha=0.7, linewidth=2)
    ax2.set_ylabel('Loss', fontsize=14)
    
    # Plot LLC measurements again
    if llc_measurements:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(llc_epochs, llc_means, 
                    color='green', label='LLC', marker='o',
                    linewidth=3, markersize=8)
        ax2_twin.set_ylabel('LLC', color='green', fontsize=14)
        ax2_twin.tick_params(axis='y', labelcolor='green')
    
    # Third subplot: Final adversarial robustness
    if adversarial_results:
        epsilons = sorted(adversarial_results.keys())
        accuracies = [adversarial_results[eps] for eps in epsilons]
        
        ax3.plot(epsilons, accuracies, 
                marker='o', markersize=10, linewidth=3,
                color='blue', label='Adversarial Accuracy')
        
        # Add points with values
        for eps, acc in zip(epsilons, accuracies):
            ax3.annotate(f'{acc:.1f}%', 
                        (eps, acc),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center', fontsize=12)

        ax3.set_xlabel('Perturbation Size (ε)', fontsize=14)
        ax3.set_ylabel('Accuracy (%)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Final Adversarial Robustness', fontsize=14)
    
    # Add legends
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper left', fontsize=12)
    if adversarial_results:
        ax3.legend(loc='upper right', fontsize=12)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_sampling_evolution(
    llc_stats: Dict[str, Any], 
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 9)
) -> plt.Figure:
    """Plot LLC loss traces from LLC estimation.
    
    This recreates the original plot_sampling_evolution from utils_outdated.py
    
    Args:
        llc_stats: Dictionary containing 'loss/trace' and 'llc_average_mean'
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    trace = llc_stats['loss/trace']
    print("Loss trace shape:", trace.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get dimensions correctly
    num_chains, num_draws = trace.shape 
    sgld_step = list(range(num_draws))
    
    # Plot individual chains
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(num_chains):
        draws = trace[i]
        color = colors[i % len(colors)]
        ax.plot(sgld_step, draws, linewidth=1.5, label=f"chain {i}", 
                color=color, alpha=0.8)

    # Plot mean with black dashed line
    mean = np.mean(trace, axis=0)
    ax.plot(sgld_step, mean, color="black", linestyle="--", 
            linewidth=3, label="mean", zorder=3)

    # Add std shading
    std = np.std(trace, axis=0)
    ax.fill_between(sgld_step, mean - std, mean + std, 
                    color="gray", alpha=0.3, zorder=2)

    # Formatting
    ax.set_title(f"Loss Trace, avg LLC = {llc_stats['llc_average_mean']:.2f}",
                fontsize=16, fontweight='bold')
    ax.set_xlabel("SGLD Step", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # Save and show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    
    return fig


def plot_llc_across_training_multiple_eps(
    llc_results: Dict[float, List[float]],
    checkpoints: List[int],
    title: str = 'LLC Evolution During Training',
    save_path: Optional[Path] = None,
    n_runs: int = 3,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot LLC evolution during training for multiple epsilon values.
    
    This recreates the original plot_llc_across_training_multiple_eps from utils_outdated.py
    
    Args:
        llc_results: Dictionary mapping epsilon values to lists of LLC values across training
        checkpoints: List of checkpoint numbers/epochs
        title: Plot title
        save_path: Path to save figure
        n_runs: Number of runs per epsilon
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = ScientificPlotStyle.COLORS
    
    # For each epsilon, average the measurements from different runs
    for i, (eps, llc_values) in enumerate(sorted(llc_results.items())):
        # Get the actual number of measurements
        n_measurements = len(llc_values)
        
        # Ensure checkpoints match the number of measurements
        actual_checkpoints = checkpoints[:n_measurements//n_runs]
        
        # Reshape the values into (n_runs, n_measurements_per_run)
        n_measurements_per_run = n_measurements // n_runs
        reshaped_values = np.array(llc_values).reshape(n_runs, n_measurements_per_run)
        
        # Calculate mean and std across runs
        mean_values = np.mean(reshaped_values, axis=0)
        std_values = np.std(reshaped_values, axis=0)
        
        color = colors[i % len(colors)]
        
        # Plot mean line with error bands
        ax.plot(actual_checkpoints, mean_values, 'o-', 
                label=f'ε={eps}', color=color,
                linewidth=ScientificPlotStyle.LINE_WIDTH, 
                markersize=ScientificPlotStyle.MARKER_SIZE)
        
        # Add error bands
        ax.fill_between(actual_checkpoints, 
                        mean_values - std_values,
                        mean_values + std_values,
                        color=color, alpha=0.2)
    
    # Apply styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel='Training Checkpoint',
        ylabel='Learning Coefficient (LLC)'
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

# ============================================================================
#  LLC-INFERENCE-TIME PLOTTING FUNCTIONS 
# ============================================================================

def plot_llc_traces(
    llc_results: Dict[float, Dict], 
    save_path: Optional[Path] = None, 
    show: bool = False,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot LLC traces for inference-time measurements.
    
    This recreates the original plot_llc_traces from utils_outdated.py
    
    Args:
        llc_results: Dictionary with epsilon -> {'trace': ..., 'mean': ...} structure
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ScientificPlotStyle.COLORS
    
    # Plot traces for each epsilon
    for i, (eps, results) in enumerate(sorted(llc_results.items())):
        color = colors[i % len(colors)]
        label = 'Clean' if eps == 0 else f'ε={eps}'
        
        # Handle both data formats
        if 'trace' in results:
            loss_traces = results['trace']
            mean_value = results['mean']
        else:
            continue
            
        # Reshape if needed
        if len(loss_traces.shape) == 1:
            loss_traces = loss_traces.reshape(1, -1)
        
        # Plot each chain's trace with transparency
        for chain_trace in loss_traces:
            ax.plot(chain_trace, alpha=0.6, linewidth=1, color=color)
    
        # Add a horizontal line for the mean value
        ax.axhline(y=mean_value, color=color, linestyle='--', 
                  alpha=0.8, linewidth=2, label=label)
    
    # Styling
    ax.set_title('LLC Traces vs Steps', fontsize=16, fontweight='bold')
    ax.set_xlabel('SGLD Steps', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    
    return fig


def plot_llc_distribution(
    llc_results: Dict[float, Dict], 
    save_path: Optional[Path] = None, 
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot LLC distribution as box plots for inference-time measurements.
    
    This recreates the original plot_llc_distribution from utils_outdated.py
    
    Args:
        llc_results: Dictionary with epsilon -> {'means': [...]} structure
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    box_data = []
    labels = []
    
    for eps in sorted(llc_results.keys()):
        if 'means' in llc_results[eps]:
            box_data.append(llc_results[eps]['means'])
            labels.append('Clean' if eps == 0 else f'ε={eps}')
    
    # Create box plot
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, 
                   showmeans=True, meanline=True)
    
    # Customize colors
    colors = ScientificPlotStyle.COLORS
    for i, patch in enumerate(bp['boxes']):
        color = colors[i % len(colors)]
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize other elements
    plt.setp(bp['medians'], color='darkred', linewidth=2)
    plt.setp(bp['means'], color='black', linewidth=2)
    plt.setp(bp['fliers'], marker='o', markerfacecolor='gray', alpha=0.6)
    
    # Styling
    ax.set_title('LLC Distribution Across Conditions', fontsize=16, fontweight='bold')
    ax.set_ylabel('LLC Value', fontsize=14)
    ax.set_xlabel('Condition', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    
    return fig


def plot_llc_mean_std(
    llc_results: Dict[float, Dict], 
    save_path: Optional[Path] = None, 
    show: bool = False,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot LLC mean vs std for inference-time measurements.
    
    This recreates the original plot_llc_mean_std from utils_outdated.py
    
    Args:
        llc_results: Dictionary with epsilon -> {'mean': ..., 'std': ...} structure
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    eps_values = sorted(llc_results.keys())
    mean_values = [llc_results[eps]['mean'] for eps in eps_values]
    std_values = [llc_results[eps]['std'] for eps in eps_values]
    
    # Plot with error bars
    ax.errorbar(eps_values, mean_values, yerr=std_values, 
               marker='o', capsize=8, capthick=2, linewidth=3,
               markersize=10, color=ScientificPlotStyle.COLORS[0])
    
    # Add value annotations
    for eps, mean, std in zip(eps_values, mean_values, std_values):
        label = 'Clean' if eps == 0 else f'ε={eps}'
        ax.annotate(f'{mean:.2f}±{std:.2f}', 
                   (eps, mean), xytext=(10, 10),
                   textcoords='offset points', fontsize=10)
    
    # Styling
    ax.set_title('LLC Mean vs Perturbation Strength', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epsilon (ε)', fontsize=14)
    ax.set_ylabel('LLC Value', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    
    return fig


# ============================================================================
# COMBINED SAE + LLC PLOTTING FUNCTIONS 
# ============================================================================

def plot_combined_sae_llc_correlation(
    sae_data: Dict[float, List[float]],
    llc_data: Dict[float, List[float]], 
    title: str = 'SAE Feature Count vs LLC Correlation',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE
) -> plt.Figure:
    """COMBINED ANALYSIS: Plot correlation between SAE feature counts and LLC values.
    
    Args:
        sae_data: Dictionary mapping epsilon to SAE feature counts
        llc_data: Dictionary mapping epsilon to LLC values
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get common epsilons
    common_eps = sorted(set(sae_data.keys()) & set(llc_data.keys()))
    
    sae_means = [np.mean(sae_data[eps]) for eps in common_eps]
    sae_stds = [np.std(sae_data[eps]) for eps in common_eps]
    llc_means = [np.mean(llc_data[eps]) for eps in common_eps]
    llc_stds = [np.std(llc_data[eps]) for eps in common_eps]
    
    # Scatter plot with error bars
    ax.errorbar(sae_means, llc_means, xerr=sae_stds, yerr=llc_stds,
               fmt='o', capsize=8, markersize=12, linewidth=2,
               color=ScientificPlotStyle.COLORS[0])
    
    # Add epsilon annotations
    for i, eps in enumerate(common_eps):
        ax.annotate(f'ε={eps}', (sae_means[i], llc_means[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12)
    
    # Styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel='SAE Feature Count',
        ylabel='LLC Value',
        legend=False
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

# ============================================================================
# PLACEHOLDER FUNCTIONS FOR CLEAN/ADVERSARIAL LLC COMPARISON
# ============================================================================

def plot_clean_vs_adversarial_llc(
    clean_llc: Dict[float, List[float]],
    adv_llc: Dict[float, List[float]], 
    title: str = 'Clean vs Adversarial LLC Analysis',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE
) -> plt.Figure:
    """Plot comparison of LLC values between clean and adversarial data.
    
    Args:
        clean_llc: Dictionary mapping epsilon to clean LLC values
        adv_llc: Dictionary mapping epsilon to adversarial LLC values
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get common epsilons
    common_eps = sorted(set(clean_llc.keys()) & set(adv_llc.keys()))
    
    # Plot clean LLC
    clean_means = [np.mean(clean_llc[eps]) for eps in common_eps]
    clean_stds = [np.std(clean_llc[eps]) for eps in common_eps]
    
    ax.errorbar(common_eps, clean_means, yerr=clean_stds,
               label='Clean Data LLC', 
               **ScientificPlotStyle.errorbar_kwargs(0))
    
    # Plot adversarial LLC
    adv_means = [np.mean(adv_llc[eps]) for eps in common_eps]
    adv_stds = [np.std(adv_llc[eps]) for eps in common_eps]
    
    ax.errorbar(common_eps, adv_means, yerr=adv_stds,
               label='Adversarial Data LLC',
               **ScientificPlotStyle.errorbar_kwargs(1))
    
    # Apply styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel='Training Epsilon (ε)',
        ylabel='LLC Value'
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig