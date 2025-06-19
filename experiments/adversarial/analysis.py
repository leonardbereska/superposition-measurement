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

def plot_llc_vs_robustness(
    llc_results: Dict[float, List[float]],
    robustness_results: Dict[float, List[float]],
    title: str = 'LLC vs Model Robustness',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE,
    color_idx: int = 0,
    add_annotations: bool = True
) -> plt.Figure:
    """Plot LLC values vs robustness
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

# ============================================================================
# IMPROVED ARCHITECTURE: SEPARATE SAE AND LLC PLOTTING
# ============================================================================

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



# ============================================================================
# UPDATED MAIN GENERATE_PLOTS FUNCTION  
# ============================================================================


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

    # LLC-specific plots (integrated from generate_llc_plots)
    if plot_args.get('plot_llc_evolution', False) or plot_args.get('plot_llc_clean_vs_adv', False):
        # Extract LLC data
        has_llc_data = any('checkpoint_llc_analysis' in results[str(eps)] for eps in epsilons if 'checkpoint_llc_analysis' in results[str(eps)])
        if has_llc_data:
            # 1. LLC evolution across training for multiple epsilons
            if plot_args.get('plot_llc_evolution', False):
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
                if llc_checkpoint_evolution:
                    max_checkpoints = max(len(values) for values in llc_checkpoint_evolution.values())
                    checkpoints = list(range(0, max_checkpoints * metadata['llc']['measure_frequency'], metadata['llc']['measure_frequency']))[:max_checkpoints]
                    fig = plot_llc_across_training_multiple_eps(
                        llc_results=llc_checkpoint_evolution,
                        checkpoints=checkpoints,
                        title="LLC Evolution During Training - Multi-Epsilon Comparison",
                        save_path=save_path("llc_evolution_multi_epsilon"),
                        n_runs=metadata['adversarial']['n_runs']
                    )
                    figures["llc_evolution_multi_epsilon"] = fig
            # 2. Clean vs Adversarial LLC comparison
            if plot_args.get('plot_llc_clean_vs_adv', False):
                clean_llc_data = {}
                adv_llc_data = {}
                for eps in epsilons:
                    eps_str = str(eps)
                    if 'llc_clean' in results[eps_str] and results[eps_str]['llc_clean']:
                        clean_llc_data[eps] = [entry['llc_mean'] for entry in results[eps_str]['llc_clean'] if entry['llc_mean'] is not None]
                    if 'llc_adversarial' in results[eps_str] and results[eps_str]['llc_adversarial']:
                        adv_llc_data[eps] = [entry['llc_mean'] for entry in results[eps_str]['llc_adversarial'] if entry['llc_mean'] is not None]
                if clean_llc_data and adv_llc_data:
                    fig = plot_clean_vs_adversarial_llc(
                        clean_llc=clean_llc_data,
                        adv_llc=adv_llc_data,
                        title=f"Clean vs Adversarial LLC - {model_type.upper()} {n_classes}-class",
                        save_path=save_path("llc_clean_vs_adversarial")
                    )
                    figures["llc_clean_vs_adversarial"] = fig

    # Combined SAE + LLC plots (integrated from generate_combined_sae_llc_plots)
    if plot_args.get('plot_combined_sae_llc', False):
        # Check if both SAE and LLC data are available
        has_sae_data = any('layers' in results[str(eps)] and any('clean_feature_count' in results[str(eps)]['layers'][layer] for layer in results[str(eps)]['layers']) for eps in epsilons)
        has_llc_data = any('checkpoint_llc_analysis' in results[str(eps)] for eps in epsilons)
        if has_sae_data and has_llc_data:
            # Extract LLC data for correlation
            llc_final_values = {}
            for eps in epsilons:
                eps_str = str(eps)
                if 'checkpoint_llc_analysis' in results[eps_str]:
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
                        if (layer_name in results[eps_str]['layers'] and 'clean_feature_count' in results[eps_str]['layers'][layer_name]):
                            feature_counts = results[eps_str]['layers'][layer_name]['clean_feature_count']
                            sae_data[eps] = [x for x in feature_counts if x is not None]
                        if eps in llc_final_values:
                            llc_data[eps] = [llc_final_values[eps]] * len(sae_data.get(eps, [1]))
                    if sae_data and llc_data:
                        fig = plot_combined_sae_llc_correlation(
                            sae_data=sae_data,
                            llc_data=llc_data,
                            title=f"SAE vs LLC Correlation - {layer_name} - {model_type.upper()} {n_classes}-class",
                            save_path=save_path(f"combined_sae_llc_correlation_{layer_name.replace('.', '_')}")
                        )
                        figures[f"combined_sae_llc_{layer_name}"] = fig

    return figures


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
    ax1.plot(all_epochs, training_history['val_acc'], 
            label='Validation Accuracy', color='blue', alpha=0.7, linewidth=2)
    ax1.plot(all_epochs, training_history['train_acc'], 
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
            
        # Convert to numpy array if it's a list
        if isinstance(loss_traces, list):
            loss_traces = np.array(loss_traces)
            
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




def plot_combined_llc_sae_evolution(
    llc_checkpoint_results: Dict[str, Any],
    sae_checkpoint_results: Dict[str, Any], 
    layer_name: str,
    epsilon: float,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Plot LLC and SAE feature count evolution side by side during training.
    
    Args:
        llc_checkpoint_results: Results from analyze_checkpoints_with_llc
        sae_checkpoint_results: Results from analyze_checkpoints_with_sae
        layer_name: Layer name for SAE analysis
        epsilon: Epsilon value to analyze
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    if title is None:
        title = f'LLC vs SAE Evolution During Training - ε={epsilon}, Layer: {layer_name}'
    
    # Extract LLC data
    llc_data = llc_checkpoint_results['epsilon_analysis'][str(epsilon)]
    llc_epochs = [entry['epoch'] for entry in llc_data]
    llc_means = [entry['llc_mean'] for entry in llc_data]
    llc_stds = [entry['llc_std'] for entry in llc_data]
    
    # Extract SAE data
    sae_layer_data = sae_checkpoint_results['epsilon_analysis'][str(epsilon)][layer_name]
    sae_epochs = [entry['epoch'] for entry in sae_layer_data if entry['feature_count'] is not None]
    sae_features = [entry['feature_count'] for entry in sae_layer_data if entry['feature_count'] is not None]
    
    # Plot LLC evolution
    ax1.errorbar(llc_epochs, llc_means, yerr=llc_stds,
                marker='o', linewidth=ScientificPlotStyle.LINE_WIDTH,
                markersize=ScientificPlotStyle.MARKER_SIZE,
                color=ScientificPlotStyle.COLORS[0],
                capsize=ScientificPlotStyle.CAPSIZE,
                capthick=ScientificPlotStyle.CAPTHICK)
    
    ax1.set_title('LLC Evolution', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS, fontweight='bold')
    ax1.set_xlabel('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax1.set_ylabel('Learning Coefficient', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax1.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax1.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    # Plot SAE feature count evolution
    ax2.plot(sae_epochs, sae_features,
            marker='s', linewidth=ScientificPlotStyle.LINE_WIDTH,
            markersize=ScientificPlotStyle.MARKER_SIZE,
            color=ScientificPlotStyle.COLORS[1])
    
    ax2.set_title(f'SAE Feature Count Evolution\n({layer_name})', 
                 fontsize=ScientificPlotStyle.FONT_SIZE_LABELS, fontweight='bold')
    ax2.set_xlabel('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax2.set_ylabel('Feature Count', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax2.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax2.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    plt.suptitle(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_aggregated_llc_sae_evolution(
    llc_data: List[Dict],
    sae_data: List[Dict],
    layer_name: str,
    epsilon: float,
    n_runs: int,
    title: str = 'Aggregated LLC vs SAE Evolution During Training',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:
    """Plot aggregated LLC vs SAE evolution across multiple runs.
    
    This gives you the comparison plot you wanted, aggregated across all runs!
    
    Args:
        llc_data: List of LLC measurement dictionaries from all runs
        sae_data: List of SAE measurement dictionaries from all runs
        layer_name: Layer name for SAE analysis
        epsilon: Epsilon value being plotted
        n_runs: Number of runs for proper averaging
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ============================================================================
    # LEFT PLOT: LLC Evolution (Aggregated across runs)
    # ============================================================================
    
    # Group LLC data by epoch
    llc_by_epoch = {}
    for entry in llc_data:
        epoch = entry['epoch']
        if epoch not in llc_by_epoch:
            llc_by_epoch[epoch] = []
        llc_by_epoch[epoch].append(entry['llc_mean'])
    
    # Calculate means and stds
    llc_epochs = sorted(llc_by_epoch.keys())
    llc_means = [np.mean(llc_by_epoch[epoch]) for epoch in llc_epochs]
    llc_stds = [np.std(llc_by_epoch[epoch]) for epoch in llc_epochs]
    
    # Plot LLC with error bars
    ax1.errorbar(llc_epochs, llc_means, yerr=llc_stds,
                marker='o', linewidth=ScientificPlotStyle.LINE_WIDTH, 
                markersize=ScientificPlotStyle.MARKER_SIZE,
                capsize=ScientificPlotStyle.CAPSIZE,
                color=ScientificPlotStyle.COLORS[0], 
                label=f'LLC (n={n_runs})')
    
    ax1.set_title(f'Learning Coefficient Evolution\nε={epsilon}', 
                 fontsize=ScientificPlotStyle.FONT_SIZE_LABELS, fontweight='bold')
    ax1.set_xlabel('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax1.set_ylabel('Learning Coefficient (LLC)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax1.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax1.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    ax1.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND)
    
    # ============================================================================
    # RIGHT PLOT: SAE Feature Count Evolution (Aggregated across runs)
    # ============================================================================
    
    # Group SAE data by epoch
    sae_by_epoch = {}
    for entry in sae_data:
        if entry['feature_count'] is not None:
            epoch = entry['epoch']
            if epoch not in sae_by_epoch:
                sae_by_epoch[epoch] = []
            sae_by_epoch[epoch].append(entry['feature_count'])
    
    # Calculate means and stds
    sae_epochs = sorted(sae_by_epoch.keys())
    sae_means = [np.mean(sae_by_epoch[epoch]) for epoch in sae_epochs]
    sae_stds = [np.std(sae_by_epoch[epoch]) for epoch in sae_epochs]
    
    # Plot SAE with error bars
    ax2.errorbar(sae_epochs, sae_means, yerr=sae_stds,
                marker='s', linewidth=ScientificPlotStyle.LINE_WIDTH,
                markersize=ScientificPlotStyle.MARKER_SIZE,
                capsize=ScientificPlotStyle.CAPSIZE,
                color=ScientificPlotStyle.COLORS[1], 
                label=f'SAE Features (n={n_runs})')
    
    ax2.set_title(f'SAE Feature Count Evolution\nε={epsilon}, {layer_name}', 
                 fontsize=ScientificPlotStyle.FONT_SIZE_LABELS, fontweight='bold')
    ax2.set_xlabel('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax2.set_ylabel('Feature Count', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax2.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax2.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    ax2.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND)
    
    plt.suptitle(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_llc_sae_dual_axis_evolution(
    llc_data: List[Dict],
    sae_data: List[Dict], 
    layer_name: str,
    epsilon: float,
    n_runs: int,
    title: str = 'LLC vs SAE Evolution - Dual Axis',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE
) -> plt.Figure:
    """Plot LLC and SAE on the same plot with dual y-axes.
    
    This version overlays both metrics to see their relationship more clearly.
    
    Args:
        llc_data: List of LLC measurement dictionaries from all runs
        sae_data: List of SAE measurement dictionaries from all runs
        layer_name: Layer name for SAE analysis
        epsilon: Epsilon value being plotted
        n_runs: Number of runs for proper averaging
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Process LLC data
    llc_by_epoch = {}
    for entry in llc_data:
        epoch = entry['epoch']
        if epoch not in llc_by_epoch:
            llc_by_epoch[epoch] = []
        llc_by_epoch[epoch].append(entry['llc_mean'])
    
    llc_epochs = sorted(llc_by_epoch.keys())
    llc_means = [np.mean(llc_by_epoch[epoch]) for epoch in llc_epochs]
    llc_stds = [np.std(llc_by_epoch[epoch]) for epoch in llc_epochs]
    
    # Plot LLC on left y-axis
    color1 = ScientificPlotStyle.COLORS[0]
    ax1.set_xlabel('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax1.set_ylabel('Learning Coefficient (LLC)', color=color1, fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    
    line1 = ax1.errorbar(llc_epochs, llc_means, yerr=llc_stds,
                        marker='o', linewidth=ScientificPlotStyle.LINE_WIDTH,
                        markersize=ScientificPlotStyle.MARKER_SIZE,
                        color=color1, label='LLC',
                        capsize=ScientificPlotStyle.CAPSIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax1.tick_params(axis='x', labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    
    # Create second y-axis for SAE data
    ax2 = ax1.twinx()
    
    # Process SAE data
    sae_by_epoch = {}
    for entry in sae_data:
        if entry['feature_count'] is not None:
            epoch = entry['epoch']
            if epoch not in sae_by_epoch:
                sae_by_epoch[epoch] = []
            sae_by_epoch[epoch].append(entry['feature_count'])
    
    sae_epochs = sorted(sae_by_epoch.keys())
    sae_means = [np.mean(sae_by_epoch[epoch]) for epoch in sae_epochs]
    sae_stds = [np.std(sae_by_epoch[epoch]) for epoch in sae_epochs]
    
    # Plot SAE on right y-axis
    color2 = ScientificPlotStyle.COLORS[1]
    ax2.set_ylabel('SAE Feature Count', color=color2, fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    
    line2 = ax2.errorbar(sae_epochs, sae_means, yerr=sae_stds,
                        marker='s', linewidth=ScientificPlotStyle.LINE_WIDTH,
                        markersize=ScientificPlotStyle.MARKER_SIZE,
                        color=color2, label='SAE Features',
                        capsize=ScientificPlotStyle.CAPSIZE)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    
    # Add combined legend
    lines = [line1, line2]
    labels = ['LLC', f'SAE Features ({layer_name})']
    ax1.legend(lines, labels, loc='upper left', fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND)
    
    # Add grid and title
    ax1.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    plt.title(f'{title}\nε={epsilon}, {layer_name}', 
             fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_llc_vs_sae_correlation_over_time(
    llc_checkpoint_results: Dict[str, Any],
    sae_checkpoint_results: Dict[str, Any],
    layer_name: str,
    epsilon: float,
    title: str = 'LLC vs SAE Correlation Over Time',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE
) -> plt.Figure:
    """Plot correlation between LLC (global) and SAE features (layer-specific) over training time.
    
    Args:
        llc_checkpoint_results: Results from analyze_checkpoints_with_llc
        sae_checkpoint_results: Results from analyze_checkpoints_with_sae
        layer_name: Layer name for SAE analysis
        epsilon: Epsilon value to analyze
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    eps_str = str(epsilon)
    
    # Extract LLC data (global model property)
    if ('epsilon_analysis' in llc_checkpoint_results and 
        eps_str in llc_checkpoint_results['epsilon_analysis']):
        
        llc_data = llc_checkpoint_results['epsilon_analysis'][eps_str]
        llc_dict = {entry['epoch']: entry['llc_mean'] for entry in llc_data}
    else:
        llc_dict = {}
    
    # Extract SAE data (layer-specific)
    if ('epsilon_analysis' in sae_checkpoint_results and 
        eps_str in sae_checkpoint_results['epsilon_analysis'] and
        layer_name in sae_checkpoint_results['epsilon_analysis'][eps_str]):
        
        sae_data = sae_checkpoint_results['epsilon_analysis'][eps_str][layer_name]
        sae_dict = {entry['epoch']: entry['feature_count'] for entry in sae_data 
                   if entry['feature_count'] is not None}
    else:
        sae_dict = {}
    
    # Get common epochs and create time series
    common_epochs = sorted(set(llc_dict.keys()) & set(sae_dict.keys()))
    
    if common_epochs:
        llc_values = [llc_dict[epoch] for epoch in common_epochs]
        sae_values = [sae_dict[epoch] for epoch in common_epochs]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(llc_values, sae_values)[0, 1] if len(llc_values) > 1 else 0
        
        # Create scatter plot with color gradient for time
        scatter = ax.scatter(sae_values, llc_values, 
                           c=common_epochs, cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
        
        # Add trend line
        z = np.polyfit(sae_values, llc_values, 1)
        p = np.poly1d(z)
        ax.plot(sae_values, p(sae_values), "--", alpha=0.8, linewidth=2, color='red')
        
        # Add correlation annotation
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title(f'{title}\nε={epsilon}, {layer_name}', 
                fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel(f'SAE Feature Count ({layer_name})', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.set_ylabel('LLC (Global Model)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig