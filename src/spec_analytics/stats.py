"""Statistics helpers: CV tables, significance annotation, and the limma
empirical-Bayes moderated t-test. Extracted from _core.py (REFACTOR_PLAN.md
step 2); behaviour unchanged. scipy/statsmodels are imported lazily inside the
functions that need them."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_cv_table(df, sample_info, level, group_col, min_values_for_cv,
                      hue_col=None):
    """
    Return long DataFrame with columns ['cv', 'group'] (and 'hue' when
    `hue_col` is set) across all replicate groups defined by
    `sample_info[group_col]`. Used by violin and stacked-bar plots.

    When `hue_col` is provided, replicates are partitioned by both `group_col`
    and `hue_col`; each (group, hue) cell with at least `min_values_for_cv`
    valid measurements contributes one CV per feature.
    """
    rows = []
    keys = ['run', 'engine']
    si_keys = [group_col] + ([hue_col] if hue_col else [])
    for partition, partition_df in sample_info.groupby(si_keys):
        if not isinstance(partition, tuple):
            partition = (partition,)
        group_value = partition[0]
        hue_value = partition[1] if hue_col else None

        sub = df.merge(partition_df[keys], on=keys, how='inner')
        if sub.empty:
            continue

        if level == 'protein':
            pg = (sub.drop_duplicates(keys + ['protein_group'])
                     .groupby('protein_group')['pg_intensity']
                     .agg(['mean', 'std', 'count']))
        elif level == 'peptide':
            pg = (sub.drop_duplicates(keys + ['peptide_id'])
                     .groupby('peptide_id')['peptide_intensity']
                     .agg(['mean', 'std', 'count']))
        elif level == 'precursor':
            pg = (sub.groupby('precursor_id')['precursor_intensity']
                     .agg(['mean', 'std', 'count']))
        else:
            raise ValueError(f'level must be protein|peptide|precursor, got {level!r}')

        pg = pg[pg['count'] >= min_values_for_cv].copy()
        pg['cv'] = pg['std'] / pg['mean']
        pg = pg[np.isfinite(pg['cv'])]
        for cv in pg['cv']:
            row = {'cv': float(cv), 'group': str(group_value)}
            if hue_col:
                row['hue'] = str(hue_value)
            rows.append(row)
    cols = ['cv', 'group'] + (['hue'] if hue_col else [])
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _annotate_significance(ax, df, x_col, y_col, *, test, correction,
                           show_ns, pairs, fontsize):
    """Run pairwise tests and draw bracket+stars on `ax`.

    Categories on the x-axis are read from `ax.get_xticklabels()` so the
    bracket positions stay aligned with whatever seaborn drew.
    """
    from itertools import combinations
    from scipy import stats as _stats

    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    if not tick_labels or any(not lab for lab in tick_labels):
        tick_labels = [str(c) for c in pd.unique(df[x_col])]
    n = len(tick_labels)
    if n < 2:
        return

    if pairs is None:
        pairs = list(combinations(range(n), 2))
    else:
        pairs = [tuple(sorted(p)) for p in pairs]

    p_values = []
    for i, j in pairs:
        a = df.loc[df[x_col].astype(str) == tick_labels[i], y_col].dropna().to_numpy()
        b = df.loc[df[x_col].astype(str) == tick_labels[j], y_col].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            p_values.append(float('nan'))
            continue
        if test == 'welch':
            _, p = _stats.ttest_ind(a, b, equal_var=False)
        elif test == 'ttest':
            _, p = _stats.ttest_ind(a, b, equal_var=True)
        elif test == 'mwu':
            _, p = _stats.mannwhitneyu(a, b, alternative='two-sided')
        else:
            raise ValueError(
                f"significance_test must be 'welch', 'ttest', or 'mwu', "
                f"got {test!r}"
            )
        p_values.append(float(p))

    if correction is not None and len([p for p in p_values if not np.isnan(p)]) > 1:
        from statsmodels.stats.multitest import multipletests
        valid_idx = [k for k, p in enumerate(p_values) if not np.isnan(p)]
        valid_p = [p_values[k] for k in valid_idx]
        _, p_adj, *_ = multipletests(valid_p, method=correction)
        for k, padj in zip(valid_idx, p_adj):
            p_values[k] = float(padj)

    def _label(p):
        if np.isnan(p):
            return None
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return 'ns' if show_ns else None

    # Stagger bracket heights above the highest data point so they don't
    # overlap. Each tier sits ~6% of yrange above the previous one.
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    base = ymax + yrange * 0.02
    h = yrange * 0.025
    tier_step = yrange * 0.07

    drawn = 0
    for k, ((i, j), p) in enumerate(zip(pairs, p_values)):
        text = _label(p)
        if text is None:
            continue
        y = base + drawn * tier_step
        ax.plot([i, i, j, j], [y, y + h, y + h, y],
                color='black', linewidth=1.0, clip_on=False)
        ax.text((i + j) / 2, y + h, text,
                ha='center', va='bottom', fontsize=fontsize,
                fontweight='bold', clip_on=False)
        drawn += 1

    if drawn:
        ax.set_ylim(ymin, base + drawn * tier_step + h * 2)


def _inv_trigamma(x):
    """Inverse of the trigamma function via Newton iteration (limma squeezeVar)."""
    from scipy.special import polygamma
    y = 0.5 + 1.0 / x
    for _ in range(50):
        t = polygamma(1, y)
        y += t * (1 - t / x) / polygamma(2, y)
    return y


def _ebayes_moderated_p(mat_a, mat_b, n_a, n_b, valid):
    """limma empirical-Bayes moderated two-sample t-test (Smyth 2004) on log2
    intensities. Borrows variance across all features via an inverse-chi-square
    prior — far more powerful than per-feature Welch when replicates are few.

    `mat_a`/`mat_b` are linear intensities with missing values as NaN; `valid`
    flags rows to use when estimating the prior. Returns (log2_fc, p_value,
    prior_df) over all rows; rows outside `valid` (or with <2 reps either side)
    get NaN p-values.
    """
    from scipy import stats
    from scipy.special import psi, polygamma
    la = np.log2(mat_a)
    lb = np.log2(mat_b)
    log2_fc = np.nanmean(la, axis=1) - np.nanmean(lb, axis=1)
    var_a = np.nanvar(la, axis=1, ddof=1)
    var_b = np.nanvar(lb, axis=1, ddof=1)
    dfree = (n_a - 1) + (n_b - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sp2 = ((n_a - 1) * var_a + (n_b - 1) * var_b) / dfree   # pooled variance

    est = valid & (dfree > 0) & np.isfinite(sp2) & (sp2 > 0)
    dfi = dfree[est].astype(float)
    z = np.log(sp2[est])
    e = z - psi(dfi / 2) + np.log(dfi / 2)
    ez = e.mean()
    var_adj = np.var(e, ddof=1) - np.mean(polygamma(1, dfi / 2))
    if var_adj <= 0:
        d0 = np.inf
        s0_2 = np.exp(ez)
    else:
        d0 = 2 * _inv_trigamma(var_adj)
        s0_2 = np.exp(ez + psi(d0 / 2) - np.log(d0 / 2))

    if np.isinf(d0):                       # prior infinitely strong -> common variance
        s2_mod = np.full_like(sp2, s0_2)
        df_tot = np.full_like(sp2, 1e6)
    else:
        s2_mod = (d0 * s0_2 + dfree * sp2) / (d0 + dfree)
        df_tot = dfree + d0
    with np.errstate(divide='ignore', invalid='ignore'):
        se = np.sqrt(s2_mod) * np.sqrt(1.0 / n_a + 1.0 / n_b)
        t = log2_fc / se
    p = 2 * stats.t.sf(np.abs(t), df=df_tot)
    p = np.where(valid & np.isfinite(t), p, np.nan)
    return log2_fc, p, d0
