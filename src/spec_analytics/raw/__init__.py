"""Raw-file analysis (manual mode) — see REFACTOR_PLAN.md §7.

Targeted extraction on vendor raw files via alpharaw: HDF caching, XICs of
specific m/z (or peptide) targets within a retention-time window, and
half-maximum peak picking. alpharaw is an optional dependency
(``pip install spec_analytics[raw]``) and is imported lazily inside
``load_ms_data`` only — the pure array helpers (``extract_xic``, ``pick_peak``,
``extract_targets``, ``peptide_mz``) need only numpy/pandas/alphabase.
"""

from __future__ import annotations

from .extraction import (  # noqa: F401
    load_ms_data,
    extract_xic,
    sum_spectra,
    pick_peak,
    points_in_window,
    integrate_peak,
    filter_spectra,
    peak_from_fragments,
    match_peaks,
    extract_targets,
    peptide_mz,
)

from .detect import (  # noqa: F401
    detect_precursors,
    pattern_scores,
)

from .metadata import (  # noqa: F401
    read_acquisition_metadata,
    survey_raw_files,
    acquisition_times,
)

from .targeted import (  # noqa: F401
    SIL_MODIFICATIONS,
    calibration_fit,
    quantify_targets,
    register_sil_modifications,
    summed_fragment_xic,
    target_fragments,
    top_n_fragments,
    y_ions_above_precursor,
)

from . import extraction  # noqa: F401
from . import detect  # noqa: F401
from . import metadata  # noqa: F401
from . import targeted  # noqa: F401
