"""spec_analytics — engine-agnostic MS-proteomics analysis library."""
from __future__ import annotations
from .schema import (
    LONG_DF_REQUIRED, LONG_DF_OPTIONAL, SAMPLE_INFO_REQUIRED,
    SAMPLE_INFO_OPTIONAL, ALPHABASE_MODS, validate_df, validate_sample_info,
)
from .proteins import (
    group_proteins_by_shared_peptides, group_proteins_by_signature,
    build_peptide_id,
)
from .sequences import (
    count_missed_cleavages, digest_protein, theoretical_coverage, gravy,
    compute_theoretical_coverage, load_protein_sequences, compute_protein_info,
)
from .stats import _compute_cv_table
from .filters import (
    filter_qvalue, filter_runs, filter_sample_info, filter_outlier_runs,
)
from .quant import compute_directlfq_pg_intensity
from .plotting import *
from .io import (
    load_experiments, detect_engine, sample_info_from_experiments,
    split_by_engine, load_peaks, load_diann,
)
