"""Plotting subpackage. Re-exports every plot_* plus init_plotting and the
PALETTE* constants; the submodule split is an internal detail."""

from __future__ import annotations

from ._style import (  # noqa: F401
    init_plotting, PALETTE, PALETTE_DOUBLE, PALETTE_SINGLE, PALETTE_CV_CATEGORIES,
    DEFAULT_PANEL_SIZE, set_default_panel_size,
    replicate_point_size, drawn_bar_width_inches, scale_replicate_points,
    set_replicate_point_scale, set_axes_size_inches,
)
from .cv import (  # noqa: F401
    plot_cv_violin, plot_cv_ecdf, plot_cv_stacked_bar_combined,
    plot_cv_vs_abundance,
)
from .bars import (  # noqa: F401
    plot_bar, plot_boxplot_with_points, plot_median_scatter, plot_overlapping_bars,
)
from .pca import plot_pca  # noqa: F401
from .rank import plot_rank  # noqa: F401
from .overlap import (  # noqa: F401
    plot_venn, plot_set_overlap, plot_upset, plot_completeness,
)
from .correlation import plot_correlation, plot_qc_protein_heatmap  # noqa: F401
from .distributions import (  # noqa: F401
    plot_intensity_histogram, plot_peptide_gravy_distribution,
    plot_peptide_length_distribution, plot_coverage_histogram,
)
from .volcano import plot_volcano  # noqa: F401
from .spectrum import plot_annotated_spectrum, ION_COLORS  # noqa: F401
from .bars import (  # noqa: F401
    BAR_PITCH_IN, BAR_WIDTH_IN, fix_bar_geometry,
)

from . import (  # noqa: F401
    _style, cv, bars, pca, rank, overlap, correlation, distributions, volcano,
    spectrum,
)
