"""Shared plotting utilities for scientific visualizations."""


class ScientificPlotStyle:
    """Standard style settings for scientific visualizations.

    Provides consistent colors, font sizes, and plot element parameters
    across all experiment visualizations.
    """

    # Color palette - soft muted colors for data series
    COLORS = ['#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']  # yellow, green, blue, purple
    ERROR_COLOR = '#BA898A'  # soft red for error indicators
    REFERENCE_LINE_COLOR = '#8B5C5D'  # dark red for reference lines

    # Typography
    FONT_SIZE_TITLE = 48
    FONT_SIZE_LABELS = 36
    FONT_SIZE_TICKS = 36
    FONT_SIZE_LEGEND = 28

    # Plot elements
    MARKER_SIZE = 15
    LINE_WIDTH = 5.0
    CAPSIZE = 12
    CAPTHICK = 5.0
    GRID_ALPHA = 0.3

    # Figure dimensions
    FIGURE_SIZE = (12, 10)
    COMBINED_FIG_SIZE = (20, 10)

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
