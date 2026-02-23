"""Analysis and visualization utilities for adversarial robustness experiments."""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_default_config
from utils import load_config
from plotting import ScientificPlotStyle

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
    
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
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
               alpha=0.8)
    
    # Apply styling
    ax.set_title(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel('adversarial training strength (ε)', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.set_ylabel('feature count ratio', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    ax.set_yscale('log')
    try:
        if plot_args['y_lim']:
            ax.set_ylim(plot_args['y_lim'])
    except Exception:
        pass
    try:
        if plot_args['y_ticks']:
            ax.set_yticks(plot_args['y_ticks'])
            ax.set_yticklabels(plot_args['y_tick_labels'])
        else:
            ax.set_yticks([0.5, 1.0, 2.0])
            ax.set_yticklabels(['0.5', '1.0', '2.0'])
    except Exception:
        pass
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