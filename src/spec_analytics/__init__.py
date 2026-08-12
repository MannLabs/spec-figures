"""
spec_analytics — engine-agnostic MS-proteomics analysis library.

Installable, submodule-based successor to the single-file ``Analytics_core.py``.
Import it once and call everything off the top-level namespace, exactly as the
old module was used::

    import spec_analytics as core

    core.init_plotting()
    df, sample_info = core.load_experiments(experiments)
    df   = core.filter_qvalue(df, 0.01)
    wide = core.to_wide(df, sample_info, level='pg_intensity')
    core.plot_pca(wide, sample_info, color_by='condition1')

The refactor (see REFACTOR_PLAN.md) has split the original single ``_core``
module into focused submodules (schema, proteins, sequences, stats, io,
filters, reshape, quant, pipeline, plotting). This re-export surface stays
stable, so call sites never change — ``import spec_analytics as core``
exposes the identical top-level API.
"""

from __future__ import annotations

# --- extracted submodules (step 2) ---------------------------------------
from .schema import (  # noqa: F401
    LONG_DF_REQUIRED, LONG_DF_OPTIONAL, SAMPLE_INFO_REQUIRED,
    SAMPLE_INFO_OPTIONAL, ALPHABASE_MODS, validate_df, validate_sample_info,
)
from .proteins import (  # noqa: F401
    group_proteins_by_shared_peptides, group_proteins_by_signature,
    build_peptide_id,
)
from .sequences import (  # noqa: F401
    count_missed_cleavages, digest_protein, theoretical_coverage, gravy,
    compute_theoretical_coverage, load_protein_sequences, compute_protein_info,
)
from .stats import _compute_cv_table  # noqa: F401

# --- extracted submodules (step 4) ---------------------------------------
from .filters import (  # noqa: F401
    filter_qvalue, filter_runs, filter_sample_info, filter_outlier_runs,
)
from .reshape import to_wide, add_combined_group  # noqa: F401
from .quant import compute_directlfq_pg_intensity  # noqa: F401
from .pipeline import process_experiment  # noqa: F401
from .species import (  # noqa: F401
    SPECIES_SUFFIXES,
    assign_species,
    expected_log2_ratios,
    plot_expected_composition,
    plot_ratio_accuracy,
    plot_species_counts,
    plot_species_cv_ecdf,
    plot_species_ratio,
    species_ratio_accuracy,
    species_ratio_table,
    sum_precursors_to_protein,
)

# --- plotting subpackage (step 5): every plot_* + init_plotting + PALETTE* -
from .plotting import *  # noqa: F401,F403

# --- io subpackage (step 3): experiment dispatch + engine loaders ---------
from .io import (  # noqa: F401
    load_experiments, detect_engine, sample_info_from_experiments,
    split_by_engine, load_peaks, load_diann,
)

# The plate-map and MS-queue helpers of the full analytics library are acquisition
# tooling — they build instrument worklists — and no figure in this paper uses
# them, so they are not part of this repository.

# Expose submodules as attributes so ``core.schema`` / ``core.io.peaks`` etc.
# work. `peaks`/`diann` remain as back-compat shims re-exporting from io.
from . import (  # noqa: F401
    schema, proteins, sequences, stats, filters, reshape, quant, pipeline,
    plotting, io, peaks, diann, species,
)

__version__ = "0.1.0"


def __getattr__(name):
    """Lazily expose the optional `raw` subpackage as ``core.raw``.

    Deferred so the heavy raw-file dependency (alpharaw) is only pulled in when
    raw-file analysis is actually used.
    """
    if name == 'raw':
        import importlib
        mod = importlib.import_module(f'{__name__}.raw')
        globals()['raw'] = mod
        return mod
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
