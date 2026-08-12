"""Canonical schema: column contracts, the AlphaBase mod vocabulary, and
validators. Extracted from _core.py (REFACTOR_PLAN.md step 2); behaviour
unchanged."""

from __future__ import annotations

import pandas as pd


LONG_DF_REQUIRED = (
    'run', 'protein_group', 'genes',
    'sequence', 'mods', 'mod_sites',
    'precursor_id', 'peptide_id',
    'precursor_intensity', 'peptide_intensity', 'pg_intensity',
    'qvalue', 'score_engine', 'engine',
)

LONG_DF_OPTIONAL = (
    'protein_names', 'charge', 'mz', 'rt',
)

# `mz` and `rt` are the PER-RUN measured values (calibrated m/z, observed apex
# retention time in minutes), not library or theoretical ones — they are what a
# raw-file lookup needs as its target.

# Three intensity tiers in the canonical schema:
#   precursor_intensity  per (peptide + mods + charge, run)  -- finest grain
#   peptide_intensity    per (peptide + mods, run)           -- sum of precursors
#   pg_intensity         per (protein_group, run)            -- protein-group quant
#
# By construction:
#   * peptide_intensity is constant across all precursor rows that share the
#     same (peptide_id, run).
#   * pg_intensity is constant across all rows that share (protein_group, run).
# Different modifications on the same sequence are different peptides
# (peptidoform-resolved).

SAMPLE_INFO_REQUIRED = (
    'run', 'file_path', 'condition1', 'condition2', 'replicate', 'engine',
)

SAMPLE_INFO_OPTIONAL = ('batch', 'group')


# ----------------------------------------------------------------------------
# AlphaBase modification vocabulary
# ----------------------------------------------------------------------------
# AlphaBase convention: mods are named '<ModName>@<AA>' (or '@Protein N-term').
# Stored as two parallel ';'-separated columns: `mods` and `mod_sites`. Sites
# are 1-based positions in `sequence`; site 0 means N-terminus.

ALPHABASE_MODS = (
    'Carbamidomethyl@C',
    'Oxidation@M',
    'Acetyl@Protein_N-term',
    'Acetyl@K',
    'Phospho@S',
    'Phospho@T',
    'Phospho@Y',
    'Deamidated@N',
    'Deamidated@Q',
)

# Engine-specific mappings live in diann.py / peaks.py.


# ============================================================================
# Validators
# ============================================================================

def validate_df(df: pd.DataFrame) -> None:
    """Raise if `df` is not a valid canonical long DataFrame."""
    missing = [c for c in LONG_DF_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'df missing required columns: {missing}')
    for col in ('mods', 'mod_sites'):
        if df[col].isna().any():
            raise ValueError(f"{col!r} contains NaN; use empty string for unmodified peptides")


def validate_sample_info(sample_info: pd.DataFrame) -> None:
    """Raise if `sample_info` is not a valid sample-metadata DataFrame.

    The primary key is (run, engine): the same raw file searched by multiple
    engines produces the same `run` name in each output, so `run` alone is
    not unique across engines.
    """
    missing = [c for c in SAMPLE_INFO_REQUIRED if c not in sample_info.columns]
    if missing:
        raise ValueError(f'sample_info missing required columns: {missing}')
    if sample_info.duplicated(['run', 'engine']).any():
        dup = sample_info[sample_info.duplicated(['run', 'engine'])][['run', 'engine']]
        raise ValueError(f'sample_info has duplicate (run, engine) rows:\n{dup}')
