"""Volcano plot (limma moderated t-test).

Extracted from _core.py (REFACTOR_PLAN.md step 5); behaviour unchanged. Heavy plotting deps (matplotlib, seaborn, scipy, sklearn) are imported lazily inside the functions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._style import (_empty_plot_with_message, _hide_top_right_spines, _filter_pivot_by_validity, _LEVEL_COLS, _resolve_panel_size)
from ..stats import _ebayes_moderated_p
from ..filters import filter_runs


def plot_volcano(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    condition_a=None,
    condition_b=None,
    fc_threshold=0.585,   # log2(1.5)
    padj_threshold=0.05,
    min_valid_per_condition=3,
    min_valid_fraction=None,
    min_valid_fraction_in_any_group=None,
    palette=None,
    highlight_genes=None,
    highlight_color='#FFD700',
    highlight_size=80,
    figsize=None,
    point_size_sig=12,
    point_size_ns=6,
    alpha_sig=0.85,
    alpha_ns=0.4,
    label_top_n=0,
    label_highlighted=True,
    label_fontsize=10,
    tick_fontsize=10,
    legend_fontsize=8,
    title_fontsize=11,
    title=None,
    method='welch',
):
    """
    Volcano plot comparing two values of `group_col` (default `condition2`) at
    the chosen quantitation level.

    Pass a single-engine df. For each feature that has >= `min_valid_per_condition`
    non-zero intensity values in BOTH conditions, computes log2 fold-change
    (A/B) and a per-feature test on log2 intensities; p-values are adjusted with
    Benjamini-Hochberg.

    Under both methods `log2_fc` is the **difference of mean log2 intensities**
    (geometric), consistent with the statistic each test is computed on, rather
    than log2 of the ratio of linear means.

    `method`:
      'welch'     (default) ordinary Welch two-sample t-test per feature.
      'moderated' empirical-Bayes variance-moderated t-test (limma, Smyth 2004).
        Borrows variance across all features via an inverse-chi-square prior;
        far more powerful with few replicates (n~3), where the per-feature
        variance estimate is too noisy for Welch + BH.

    Parameters:
      level: 'precursor' | 'peptide' | 'protein'  (default 'protein').
        At non-protein levels the row id column is named after the level
        (`peptide_id` or `precursor_id`); the gene column still resolves via
        the feature's parent protein_group, so `highlight_genes=` works at any
        level (highlighting every feature of the matching gene).
      min_valid_per_condition: integer count of non-zero replicates required
        in BOTH compared conditions for the t-test (default 3, mathematically
        the minimum for a Welch test).
      min_valid_fraction: optional overall-fraction filter applied across the
        runs of the two compared conditions before the t-test.
      min_valid_fraction_in_any_group: optional per-group fraction filter
        (over `group_col`); a feature passes if at least one group hits this
        fraction. Useful for many-condition datasets — filters out features
        that are unreliable everywhere even if they squeak past the
        condition-A/B count requirement.

    All filters combine as AND.

    Returns (fig, ax, volcano_df). volcano_df columns:
      <id_col>, gene, mean_a, mean_b, log2_fc, p_value, p_adj,
      neg_log10_padj, n_a, n_b, significance ('Up' | 'Down' | 'NS'),
      highlighted (bool).
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    if level not in _LEVEL_COLS:
        raise ValueError(
            f"level must be one of {sorted(_LEVEL_COLS)}, got {level!r}"
        )
    id_col, intensity_col = _LEVEL_COLS[level]

    if df['engine'].nunique() > 1:
        raise ValueError('plot_volcano: df contains multiple engines; '
                         'use core.split_by_engine first')
    if group_col not in sample_info.columns:
        raise ValueError(f'group_col {group_col!r} missing from sample_info')

    # Pick conditions if not provided.
    available = list(sample_info[group_col].drop_duplicates())
    if condition_a is None or condition_b is None:
        if len(available) != 2:
            raise ValueError(
                f'sample_info[{group_col!r}] has {len(available)} unique values '
                f'{available!r}; pass condition_a=, condition_b=  explicitly'
            )
        if condition_a is None:
            condition_a = available[0]
        if condition_b is None:
            condition_b = available[1]

    runs_a = sample_info.loc[sample_info[group_col] == condition_a, 'run'].tolist()
    runs_b = sample_info.loc[sample_info[group_col] == condition_b, 'run'].tolist()
    if not runs_a or not runs_b:
        raise ValueError('one or both conditions matched zero runs')

    df = filter_runs(df, sample_info)
    pivot = df.pivot_table(
        index=id_col, columns='run',
        values=intensity_col, aggfunc='first',
    )
    cols_a = [r for r in runs_a if r in pivot.columns]
    cols_b = [r for r in runs_b if r in pivot.columns]

    # Apply optional fraction filters (overall + per-group). The per-condition
    # count requirement (min_valid_per_condition) is enforced inside the t-test
    # loop below.
    if min_valid_fraction is not None or min_valid_fraction_in_any_group is not None:
        compared_runs = cols_a + cols_b
        pivot_compared = pivot[compared_runs]
        pivot_compared = _filter_pivot_by_validity(
            pivot_compared, sample_info,
            min_valid_fraction=min_valid_fraction,
            min_valid_fraction_in_any_group=min_valid_fraction_in_any_group,
            group_col=group_col,
            runs_axis='columns',
            log_prefix=f'[volcano:{level}]',
        )
        pivot = pivot.loc[pivot_compared.index]

    # Map id_col -> first non-empty gene for labels (gene info travels per row
    # at every level — derived from the parent protein_group).
    gene_map = (df.assign(_g=df['genes'].astype(str).str.split(';').str[0])
                  .groupby(id_col)['_g'].agg(
                      lambda s: next((v for v in s if v), '')))

    # Vectorized Welch t-test over all features at once. Verified bit-identical
    # to the per-row loop on the test data; ~4-5x faster at precursor / peptide
    # scale where the feature count is in the tens of thousands.
    mat_a = pivot[cols_a].astype(float).to_numpy()
    mat_b = pivot[cols_b].astype(float).to_numpy()
    # Treat zero as missing — keeps log2 finite and matches the loop's filter.
    mat_a = np.where(mat_a > 0, mat_a, np.nan)
    mat_b = np.where(mat_b > 0, mat_b, np.nan)

    n_a = np.sum(~np.isnan(mat_a), axis=1)
    n_b = np.sum(~np.isnan(mat_b), axis=1)
    enough = (n_a >= min_valid_per_condition) & (n_b >= min_valid_per_condition)

    if method not in ('welch', 'moderated'):
        raise ValueError(f"method must be 'welch' or 'moderated', got {method!r}")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_a = np.nanmean(mat_a, axis=1)
        mean_b = np.nanmean(mat_b, axis=1)
        if method == 'moderated':
            # empirical-Bayes moderated t (limma); log2_fc = mean-of-log diff,
            # consistent with the test statistic.
            log2_fc, p_val, _prior_df = _ebayes_moderated_p(
                mat_a, mat_b, n_a, n_b, enough)
        else:
            # log2_fc is the difference of mean log2 intensities (geometric),
            # matching the statistic the Welch test below is computed on. It
            # used to be log2(mean_a / mean_b) — log of the linear mean — which
            # left the plotted effect size and the p-value measuring slightly
            # different things.
            log2_fc = (np.nanmean(np.log2(mat_a), axis=1)
                       - np.nanmean(np.log2(mat_b), axis=1))
            # axis=1 + nan_policy='omit' applies a per-row Welch test, omitting
            # NaNs independently in each row.
            t_result = stats.ttest_ind(np.log2(mat_a), np.log2(mat_b),
                                       axis=1, equal_var=False,
                                       nan_policy='omit')
            p_val = np.asarray(t_result.pvalue)

    ids = pivot.index.to_numpy()
    keep = enough & np.isfinite(p_val) & np.isfinite(log2_fc)
    if not keep.any():
        warnings.warn(
            f'plot_volcano: no {level}-level features passed validity '
            f'filters (min_valid_per_condition={min_valid_per_condition}). '
            f'Either too few replicates per condition or too sparse data. '
            f'Rendering an empty plot.'
        )
        fig, ax = _empty_plot_with_message(
            f'volcano: no features passed validity filters\n'
            f'(min_valid_per_condition={min_valid_per_condition})',
            figsize=_resolve_panel_size(figsize),
            title=title or f'Volcano: {condition_a} vs {condition_b}')
        empty = pd.DataFrame(columns=[
            id_col, 'gene', 'mean_a', 'mean_b', 'log2_fc', 'p_value',
            'p_adj', 'neg_log10_padj', 'n_a', 'n_b',
            'significance', 'highlighted',
        ])
        return fig, ax, empty

    kept_ids = ids[keep]
    volcano_df = pd.DataFrame({
        id_col: kept_ids,
        'gene': [gene_map.get(i, '') for i in kept_ids],
        'mean_a': mean_a[keep],
        'mean_b': mean_b[keep],
        'log2_fc': log2_fc[keep],
        'p_value': p_val[keep],
        'n_a': n_a[keep].astype(int),
        'n_b': n_b[keep].astype(int),
    })

    _, p_adj, *_ = multipletests(volcano_df['p_value'], method='fdr_bh')
    volcano_df['p_adj'] = p_adj
    volcano_df['neg_log10_padj'] = -np.log10(p_adj)

    def classify(r):
        if r['p_adj'] < padj_threshold:
            if r['log2_fc'] > fc_threshold:
                return 'Up'
            if r['log2_fc'] < -fc_threshold:
                return 'Down'
        return 'NS'
    volcano_df['significance'] = volcano_df.apply(classify, axis=1)

    if highlight_genes:
        upper = {g.upper() for g in highlight_genes}
        volcano_df['highlighted'] = volcano_df['gene'].fillna('').astype(str).str.upper().isin(upper)
    else:
        volcano_df['highlighted'] = False

    palette = palette if palette is not None else ['#E74C3C', '#3498DB']
    colors = {'Up': palette[0], 'Down': palette[1], 'NS': '#CCCCCC'}
    counts = volcano_df['significance'].value_counts()

    fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    for cat in ('NS', 'Down', 'Up'):
        sub = volcano_df[(volcano_df['significance'] == cat) & (~volcano_df['highlighted'])]
        if sub.empty:
            continue
        # Significant dots get a thin black edge so they read clearly against
        # the grey NS background and the white page; NS stay edgeless so the
        # backdrop doesn't dominate.
        is_sig = cat != 'NS'
        ax.scatter(
            sub['log2_fc'], sub['neg_log10_padj'],
            c=colors[cat],
            s=point_size_ns if cat == 'NS' else point_size_sig,
            alpha=alpha_ns if cat == 'NS' else alpha_sig,
            label=f'{cat} (n={int(counts.get(cat, 0))})',
            edgecolors='black' if is_sig else 'none',
            linewidth=0.4 if is_sig else 0,
        )
    if highlight_genes:
        sub = volcano_df[volcano_df['highlighted']]
        if not sub.empty:
            ax.scatter(
                sub['log2_fc'], sub['neg_log10_padj'],
                c=highlight_color, s=highlight_size, alpha=1.0,
                label=f'Highlighted (n={len(sub)})',
                edgecolors='black', linewidth=1, zorder=10,
            )
    ax.axhline(-np.log10(padj_threshold), color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(fc_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(-fc_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # mathtext subscripts render in the matching DejaVu math font (the default),
    # so log2 / log10 get proper subscripts and the PDF stays single-font.
    ax.set_xlabel(f'log₂(fold change)  [{condition_a} / {condition_b}]',
                  fontsize=label_fontsize)
    ax.set_ylabel('−log₁₀(adjusted p-value)', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if title is None:
        title = f'Volcano: {condition_a} vs {condition_b}'
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=8)
    ax.legend(loc='upper right', fontsize=legend_fontsize, frameon=False)

    if label_highlighted and highlight_genes:
        for _, row in volcano_df[volcano_df['highlighted']].iterrows():
            ax.annotate(
                str(row['gene']) or str(row[id_col]),
                (row['log2_fc'], row['neg_log10_padj']),
                fontsize=9, fontweight='bold',
                ha='left' if row['log2_fc'] > 0 else 'right', va='bottom',
                xytext=(8 if row['log2_fc'] > 0 else -8, 5),
                textcoords='offset points',
            )
    if label_top_n > 0:
        sig = volcano_df[(volcano_df['significance'] != 'NS') & (~volcano_df['highlighted'])]
        for _, row in sig.nsmallest(label_top_n, 'p_adj').iterrows():
            label = str(row['gene']) or str(row[id_col])
            ax.annotate(
                label, (row['log2_fc'], row['neg_log10_padj']),
                fontsize=8, alpha=0.8,
                ha='left' if row['log2_fc'] > 0 else 'right', va='bottom',
                xytext=(5 if row['log2_fc'] > 0 else -5, 5),
                textcoords='offset points',
            )

    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax, volcano_df
