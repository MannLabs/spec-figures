"""Plotting subpackage. Re-exports every plot_* plus init_plotting and the"""
from __future__ import annotations
from ._style import (
    init_plotting, PALETTE, PALETTE_DOUBLE, PALETTE_SINGLE, PALETTE_CV_CATEGORIES,
    DEFAULT_PANEL_SIZE, set_default_panel_size,
    replicate_point_size, drawn_bar_width_inches, scale_replicate_points,
    set_replicate_point_scale, set_axes_size_inches,
)
from .cv import (
    plot_cv_violin, plot_cv_ecdf, plot_cv_stacked_bar_combined,
    plot_cv_vs_abundance,
)
from .bars import (
    plot_bar, plot_boxplot_with_points, plot_median_scatter, plot_overlapping_bars,
    plot_grouped_bars,
)
from .pca import plot_pca
from .correlation import plot_correlation, plot_qc_protein_heatmap
from .volcano import plot_volcano
from .bars import (
    BAR_PITCH_IN, BAR_WIDTH_IN, BAR_GROUP_GAP_IN, fix_bar_geometry,
    set_bar_geometry,
)
from . import (
    _style, cv, bars, pca, correlation, volcano,
)
