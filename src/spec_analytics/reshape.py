"""Long<->wide reshaping and combined-group helpers. Extracted from _core.py
(REFACTOR_PLAN.md step 4); behaviour unchanged."""

from __future__ import annotations

import pandas as pd


def to_wide(
    df: pd.DataFrame,
    *,
    value: str = 'pg_intensity',
    index: str = 'run',
    columns: str = 'protein_group',
    aggfunc: str = 'first',
) -> pd.DataFrame:
    """
    Pivot the canonical long df to wide format.

    Default: index=run, columns=protein_group, values=pg_intensity. Using 'first'
    is safe because pg_intensity is constant per (protein_group, run) by
    construction.
    """
    return df.pivot_table(index=index, columns=columns, values=value, aggfunc=aggfunc)


def add_combined_group(df, sample_info, *, cols, sep=' / ', new_col='condition_combined'):
    """Add a combined group column to both `df` and `sample_info` from multiple
    columns of `sample_info`, joined with `sep`.

    Useful for the comparison plots (`plot_volcano`, `plot_venn`,
    `plot_correlation`, `plot_qc_protein_heatmap`) when you want to compare
    not just `condition2` but the cross-product, e.g. "Astral5 / 100SPD" vs
    "Astral2 / 100SPD". After this call you can pass `group_col=new_col`.

    `df` and `sample_info` are not modified in place — copies are returned.
    """
    si = sample_info.copy()
    si[new_col] = si[cols[0]].astype(str)
    for c in cols[1:]:
        si[new_col] = si[new_col] + sep + si[c].astype(str)
    keys = ['run', 'engine'] if 'engine' in df.columns and 'engine' in si.columns else ['run']
    df_out = df.merge(si[keys + [new_col]], on=keys, how='left')
    return df_out, si
