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
    # COLORS = [
    #     '#F0BE5E',  # yellow
    #     '#94B9A3',  # green
    #     '#88A7B2',  # blue
    #     '#DDC6E1',  # purple
    #     '#E8A87C',  # orange
    #     '#C38D9E',  # mauve
    #     '#41B3A3',  # teal
    #     '#8D5B4C',  # brown
    #     '#5D576B',  # dark purple
    #     '#767B91',  # slate blue
    #     '#85C7F2',  # light blue
    #     '#D64933'   # red
    # ]
    # Color palette - sorted to form a rainbow
    # COLORS = [
    #     '#BA898A',  # red
    #     '#F0BE5E',  # yellow
    #     '#94B9A3',  # green
    #     '#88A7B2',  # blue
    #     '#757F8D',  # slate
    #     '#B894B1'   # purple
    # ]
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

# def plot_robustness_curve(
#     results: Dict[float, List[float]],
#     title: str = 'Adversarial Robustness',
#     save_path: Optional[Path] = None,
#     figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
#     color_idx: int = 0
# ) -> plt.Figure:
#     """Plot robustness curve across different epsilon values.
    
#     Args:
#         results: Dictionary mapping epsilon values to list of accuracies
#         title: Plot title
#         save_path: Path to save figure
#         figsize: Figure size
#         color_idx: Index for color selection from ScientificPlotStyle
        
#     Returns:
#         Matplotlib figure
#     """
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Calculate mean and std for each epsilon
#     epsilons = sorted(results.keys())
#     means = [np.mean(results[eps]) for eps in epsilons]
#     stds = [np.std(results[eps]) for eps in epsilons]
    
#     # Get style parameters
#     style_kwargs = ScientificPlotStyle.errorbar_kwargs(color_idx)
    
#     # Plot
#     ax.errorbar(
#         epsilons, means, yerr=stds,
#         **style_kwargs
#     )
    
#     # Apply scientific plot styling
#     ScientificPlotStyle.apply_axis_style(
#         ax=ax,
#         title=title,
#         xlabel='Perturbation Size (ε)',
#         ylabel='Accuracy (%)',
#         legend=False
#     )
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
#     return fig


# def plot_combined_feature_counts(
#     results: Dict[str, Dict[float, List[float]]],
#     title: str = 'Feature Counts vs Adversarial Training Strength',
#     save_path: Optional[Path] = None,
#     figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE
# ) -> plt.Figure:
#     """Plot all feature counts (clean, adversarial, mixed) on one plot.
    
#     Args:
#         results: Dictionary with feature count results
#         title: Plot title
#         save_path: Path to save figure
#         figsize: Figure size
        
#     Returns:
#         Matplotlib figure
#     """
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Labels for the legend
#     labels = {
#         'clean_feature_count': 'Clean Inputs',
#         'adversarial_feature_count': 'Adversarial Inputs',
#         'mixed_feature_count': 'Mixed Inputs'
#     }
    
#     # Get set of all epsilons across all metrics
#     all_epsilons = set()
#     for metric in results:
#         if metric.endswith('_feature_count'):
#             all_epsilons.update(results[metric].keys())
    
#     # Sort epsilons
#     all_epsilons = sorted(all_epsilons)
    
#     # Plot each feature count metric
#     for i, metric_name in enumerate(['clean_feature_count', 'adversarial_feature_count', 'mixed_feature_count']):
#         if metric_name not in results:
#             continue
            
#         metric_data = results[metric_name]
        
#         # Calculate mean and std for each epsilon
#         epsilons = sorted(metric_data.keys())
#         means = [np.mean(metric_data[eps]) for eps in epsilons]
#         stds = [np.std(metric_data[eps]) for eps in epsilons]
        
#         # Plot with error bars using ScientificPlotStyle
#         ax.errorbar(
#             epsilons, means, yerr=stds,
#             label=labels[metric_name],
#             **ScientificPlotStyle.errorbar_kwargs(i)
#         )
    
#     # Apply scientific plot styling
#     ScientificPlotStyle.apply_axis_style(
#         ax=ax,
#         title=title,
#         xlabel='Adversarial Training Strength (ε)',
#         ylabel='Feature Count'
#     )
    
#     plt.tight_layout()
    
#     # Save if path provided
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
#     return fig


# def plot_feature_distribution_matrix(
#     results: Dict[str, Dict[float, List[float]]],
#     epsilon: float,
#     title: str = 'Feature Distribution Matrix',
#     save_path: Optional[Path] = None,
#     figsize: Tuple[int, int] = (8, 6)
# ) -> plt.Figure:
#     """Plot a heatmap of feature counts for a specific epsilon value.
    
#     Args:
#         results: Dictionary with feature count results
#         epsilon: Epsilon value to visualize
#         save_path: Path to save figure
#         figsize: Figure size
        
#     Returns:
#         Matplotlib figure
#     """
#     # Get feature counts for this epsilon
#     labels = ['Clean', 'Adversarial', 'Mixed']
#     feature_counts = []
    
#     for metric in ['clean_feature_count', 'adversarial_feature_count', 'mixed_feature_count']:
#         if metric in results and epsilon in results[metric]:
#             mean_count = np.mean(results[metric][epsilon])
#             feature_counts.append(mean_count)
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Create data matrix (just a 1D array in this case)
#     data = np.array(feature_counts).reshape(-1, 1)
    
#     # Create heatmap
#     im = ax.imshow(data, cmap='YlGnBu')
    
#     # Add colorbar
#     cbar = ax.figure.colorbar(im, ax=ax)
#     cbar.ax.set_ylabel('Feature Count', rotation=-90, va="bottom", fontsize=12)
    
#     # Show all ticks and label them
#     ax.set_yticks(np.arange(len(labels)))
#     ax.set_yticklabels(labels, fontsize=12)
#     ax.set_xticks([])  # No x-ticks needed
    
#     # Add feature count text in each cell
#     for i in range(len(labels)):
#         ax.text(0, i, f"{feature_counts[i]:.2f}", 
#                 ha="center", va="center", color="black", fontsize=14)
    
#     # Set title and adjust layout
#     ax.set_title(f'Feature Distribution (ε={epsilon})', fontsize=16)
#     fig.tight_layout()
    
#     # Save if path provided
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
    
#     return fig
# # %%
# def plot_feature_counts(
#     results: Dict[float, List[float]],
#     title: str = 'Feature Count vs Adversarial Training Strength',
#     save_path: Optional[Path] = None,
#     figsize: Tuple[int, int] = (8, 6),
#     color: str = '#88A7B2'
# ) -> plt.Figure:
#     """Plot feature counts across different epsilon values.
    
#     Args:
#         results: Dictionary mapping epsilon values to list of feature counts
#         title: Plot title
#         save_path: Path to save figure
#         figsize: Figure size
#         color: Line color
        
#     Returns:
#         Matplotlib figure
#     """
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Calculate mean and std for each epsilon
#     epsilons = sorted(results.keys())
#     means = [np.mean(results[eps]) for eps in epsilons]
#     stds = [np.std(results[eps]) for eps in epsilons]
    
#     # Plot
#     ax.errorbar(
#         epsilons, means, yerr=stds, fmt='o-',
#         color=color, linewidth=2, capsize=5, markersize=8
#     )
    
#     ax.set_xlabel('Adversarial Training Strength (ε)', fontsize=14)
#     ax.set_ylabel('Feature Count', fontsize=14)
#     ax.set_title(title, fontsize=16)
#     ax.grid(True, alpha=0.3)
    
#     # Set tick parameters
#     ax.tick_params(labelsize=12)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
    
#     return fig

# # %%
# def plot_feature_vs_robustness(
#     feature_results: Dict[float, List[float]],
#     robustness_results: Dict[float, List[float]],
#     title: str = 'Feature Count vs Model Robustness',
#     save_path: Optional[Path] = None,
#     figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
#     color_idx: int = 0,
#     add_annotations: bool = True
# ) -> plt.Figure:
#     """Plot feature counts vs robustness.
    
#     Args:
#         feature_results: Dictionary mapping epsilon values to list of feature counts
#         robustness_results: Dictionary mapping epsilon values to list of accuracies
#         title: Plot title
#         save_path: Path to save figure
#         figsize: Figure size
#         color_idx: Index for color selection from ScientificPlotStyle
#         add_annotations: Whether to add epsilon annotations
        
#     Returns:
#         Matplotlib figure
#     """
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Calculate mean and std for each epsilon
#     epsilons = sorted(set(feature_results.keys()) & set(robustness_results.keys()))
#     feature_means = [np.mean(feature_results[eps]) for eps in epsilons]
#     feature_stds = [np.std(feature_results[eps]) for eps in epsilons]
#     robustness_means = [np.mean(robustness_results[eps]) for eps in epsilons]
#     robustness_stds = [np.std(robustness_results[eps]) for eps in epsilons]
    
#     # Get style parameters
#     style_kwargs = ScientificPlotStyle.errorbar_kwargs(color_idx)
    
#     # Plot
#     ax.errorbar(
#         feature_means, robustness_means, 
#         xerr=feature_stds, yerr=robustness_stds,
#         fmt='.', capsize=ScientificPlotStyle.CAPSIZE, 
#         markersize=ScientificPlotStyle.MARKER_SIZE,
#         color=style_kwargs['color'],
#         elinewidth=style_kwargs['linewidth'],
#         capthick=ScientificPlotStyle.CAPTHICK
#     )
    
#     # Add epsilon annotations
#     if add_annotations:
#         for i, eps in enumerate(epsilons):
#             ax.annotate(
#                 f'ε={eps}', 
#                 (feature_means[i], robustness_means[i]),
#                 xytext=(10, -25), 
#                 textcoords='offset points', 
#                 color=style_kwargs['color'], 
#                 fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND
#             )
    
#     # Apply scientific plot styling
#     ScientificPlotStyle.apply_axis_style(
#         ax=ax,
#         title=title,
#         xlabel='Feature Count',
#         ylabel='Average Robustness',
#         legend=False
#     )
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
#     return fig

def generate_plots(results: Dict, output_dir: Path, results_dir: Path, plot_args: Optional[Dict[str, Any]] = None) -> Dict[str, plt.Figure]:
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
    attack_type = metadata['adversarial']['attack_type']
    experiment_name = f"{model_type.upper()} {n_classes}-class {attack_type.upper()}"  

    def save_path(name):
        experiment_string = f"{model_type}_{n_classes}-class_{attack_type}"
        return output_dir / f"{experiment_string}_{name}.pdf"
    
    epsilons = sorted([float(eps) for eps in results.keys()])
    
    layers = results[str(epsilons[0])]['layers'].keys()
    
    # Generate comprehensive overview plot with all layers and distributions
    if plot_args['plot_feature_counts']:
        fig = plot_feature_counts(
            results, layers, epsilons,
            title=f"{experiment_name}",
            save_path=save_path("feature_counts"),
            plot_args=plot_args
        )
        figures["feature_counts"] = fig

    # Generate normalized feature counts plot
    if plot_args['plot_normalized_feature_counts']:
        fig = plot_normalized_feature_counts(
            results, layers, epsilons,
            title=f"{experiment_name}",
            save_path=save_path("feature_counts_normalized"),
            plot_args=plot_args
        )
        figures["feature_counts_normalized"] = fig

    # Generate comprehensive robustness overview plot
    if plot_args['plot_robustness']:
        fig = plot_robustness(
            results, epsilons,
            title=f"{experiment_name}",
            save_path=save_path("robustness"),
            plot_args=plot_args
        )
        figures["robustness"] = fig

    return figures
  
def plot_feature_counts(
    results: Dict, 
    layers: List[str], 
    epsilons: List[float], 
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
    plot_args: Optional[Dict[str, Any]] = None
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
    if plot_args['plot_adversarial']:
        distributions = {
            "clean_feature_count": {"linestyle": "-", "name": "clean"},
            "adv_feature_count": {"linestyle": "--", "name": "adversarial"}
        }
    else:
        distributions = {
            "clean_feature_count": {"linestyle": "-", "name": "clean"}
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
            # if only one layer, don't show the layer name
            if len(layers) == 1:
                label = f"{dist_props['name']}"
            else:
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
    ax.set_xlabel('adversarial training strength (ε)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.set_ylabel('feature count', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    # Limit the number of x-ticks to 3-4 for cleaner display
    # if len(epsilons) > 4:
        # Calculate step size to get 3-4 ticks
    step = max(1, len(epsilons) // 3)
    tick_positions = epsilons[::step]
    # Ensure the last tick is included
    if epsilons[-1] not in tick_positions:
        tick_positions = np.append(tick_positions, epsilons[-1])
    ax.set_xticks(tick_positions)
    
    # Adjust legend based on number of entries
    if plot_args['plot_legend']:
        if line_count > 8:
            # Use two-column legend for many entries
            legend = ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND * 0.8, 
                    loc='best', ncol=2, framealpha=plot_args['legend_alpha'])
        else:
            legend = ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND, 
                    loc='best', framealpha=plot_args['legend_alpha'])
    
    # Set legend background to be fully opaque
    # legend.get_frame().set_alpha(1.0)
    # legend.get_frame().set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_normalized_feature_counts(
    results: Dict, 
    layers: List[str], 
    epsilons: List[float], 
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
    plot_args: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """Plot feature counts normalized by the baseline (ε=0.0) values.
    
    Args:
        results: Nested results dictionary
        layers: List of layer names
        epsilons: List of epsilon values
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        plot_args: Plotting arguments
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define line styles for different data distributions
    if plot_args['plot_adversarial']:
        distributions = {
            "clean_feature_count": {"linestyle": "-", "name": "clean"},
            "adv_feature_count": {"linestyle": "--", "name": "adversarial"}
        }
    else:
        distributions = {
            "clean_feature_count": {"linestyle": "-", "name": "clean"}
        }
    
    # Plot each layer and distribution combination
    # No markers will be used for the plot
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    # marker_styles = [None] * 9  # Create a list of None values instead of marker symbols
    line_count = 0
    
    for i, layer_name in enumerate(layers):
        for j, (dist_key, dist_props) in enumerate(distributions.items()):
            # Extract data for this layer and distribution
            layer_data = {}
            for eps in epsilons:
                layer_data[eps] = results[str(eps)]['layers'][layer_name][dist_key]
            
            # Get baseline (ε=0.0) values for normalization
            baseline_values = layer_data[0.0]  # Assuming 0.0 is always in epsilons
            baseline_mean = np.mean(baseline_values)
            
            # Calculate normalized means and propagated uncertainties
            normalized_means = []
            normalized_stds = []
            
            for eps in epsilons:
                values = np.array(layer_data[eps])
                
                # Normalize each individual measurement by the baseline mean
                normalized_values = values / baseline_mean
                
                # Calculate statistics of normalized values
                norm_mean = np.mean(normalized_values)
                norm_std = np.std(normalized_values)
                
                normalized_means.append(norm_mean)
                normalized_stds.append(norm_std)
            
            # Determine color and marker
            color_idx = i % len(ScientificPlotStyle.COLORS)
            marker = marker_styles[i % len(marker_styles)]
            
            # Create label
            if len(layers) == 1:
                label = f"{dist_props['name']}"
            else:
                label = f"{layer_name} - {dist_props['name']}"
            
            # Plot
            ax.errorbar(
                epsilons, normalized_means, yerr=normalized_stds,
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
    
    # Add horizontal reference line at y=1.0 (baseline)
    ax.axhline(y=1.0, color=ScientificPlotStyle.REFERENCE_LINE_COLOR, 
               linestyle='--', linewidth=ScientificPlotStyle.LINE_WIDTH, 
               alpha=0.8)#, label='baseline (ε=0)')
    
    # Apply styling
    ax.set_title(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel('adversarial training strength (ε)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.set_ylabel('feature count ratio', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    # Set fixed y-axis limits between 0.2 and 3
    ax.set_yscale('log')
    # ax.set_yscale('linear')
    try:
        if plot_args['y_lim']:
            ax.set_ylim(plot_args['y_lim'])
    except:
        pass
    # ax.set_ylim(0.3, 3.)
    # ax.set_ylim(0.15, 7.)
    # set yticks to 0.5, 1.0, 2.0
    # Explicitly set y-ticks to ensure no other ticks appear
    try:
        if plot_args['y_ticks']:
            ax.set_yticks(plot_args['y_ticks'])
            ax.set_yticklabels(plot_args['y_tick_labels'])
        else:
            ax.set_yticks([0.5, 1.0, 2.0])
            ax.set_yticklabels(['0.5', '1.0', '2.0'])
    except:
        pass
    # Remove minor ticks which might be showing in log scale
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    # Limit the number of x-ticks for cleaner display
    step = max(1, len(epsilons) // 3)
    tick_positions = epsilons[::step]
    if epsilons[-1] not in tick_positions:
        tick_positions = np.append(tick_positions, epsilons[-1])
    ax.set_xticks(tick_positions)
    
    # Adjust legend based on number of entries
    if plot_args['plot_legend']:
        if line_count > 8:
            legend = ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND * 0.8, 
                    loc='best', ncol=2, framealpha=plot_args['legend_alpha'])
        else:
            legend = ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND, 
                    loc='best', framealpha=plot_args['legend_alpha'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

def plot_layer_comparison(
    results: Dict, 
    layers: List[str], 
    epsilons: List[float], 
    metric_key: str, 
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
    plot_args: Optional[Dict[str, Any]] = None
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
        xlabel='adversarial training strength (ε)',
        ylabel='feature count'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_robustness(
    results: Dict, 
    epsilons: List[float], 
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE, 
    plot_args: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """Plot comprehensive overview of accuracy across all test epsilons and training epsilons.
    
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
        label = f"test ε={test_eps}"
        
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
    # y lim 0, 100
    ax.set_ylim(0, 104)

    # Apply styling with adjusted legend
    ax.set_title(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel('training epsilon (ε)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.set_ylabel('accuracy (%)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    # Adjust legend based on number of entries
    if plot_args['plot_legend']:
        if len(test_epsilons) > 8:
            # Use two-column legend for many entries
            ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND * 0.8, 
                    loc='best', ncol=2, framealpha=plot_args['legend_alpha'])
        else:
            ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND, loc='best', framealpha=plot_args['legend_alpha'])
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig