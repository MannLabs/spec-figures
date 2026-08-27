"""Row/column and run-level filtering of the canonical DataFrame. Extracted"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
def filter_qvalue(df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    """
    Keep only rows with `qvalue < threshold`. Pass-through when threshold is None.
    Records the threshold in `df.attrs['qvalue_filter']`.
    """
    if threshold is None:
        df = df.copy()
        df.attrs['qvalue_filter'] = None
        return df
    n_before = len(df)
    out = df[df['qvalue'] < threshold].copy()
    out.attrs['qvalue_filter'] = threshold
    pct = 100 * len(out) / n_before if n_before else 0
    print(f'qvalue<{threshold}: kept {len(out):,} of {n_before:,} rows ({pct:.1f}%)')
    return out
def filter_runs(df: pd.DataFrame, sample_info: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose (run, engine) appears in sample_info."""
    keys = sample_info[['run', 'engine']].drop_duplicates()
    keys['_keep'] = True
    out = df.merge(keys, on=['run', 'engine'], how='left')
    out = out[out['_keep'] == True].drop(columns='_keep')
    return out.reset_index(drop=True)
def filter_sample_info(sample_info, **conditions):
    """Subset `sample_info` by exact-match conditions on its columns.
    Useful for the load-once / filter-per-plot pattern: load all data once
    via `load_experiments(...)`, then for individual plots pass a filtered
    sample_info to restrict which runs are drawn — every plot function
    inner-joins df against sample_info on `run`, so a smaller sample_info
    naturally restricts the plot.
    Pass a single value for exact match, or a list/tuple/set for OR-matching
    several values per column. Multiple keyword filters combine as AND.
    Raises `ValueError` immediately if any filter value isn't present in the
    target column — saves time debugging silent-empty subsets caused by typos.
    >>> si_100 = filter_sample_info(sample_info, condition2='100SPD 200 ng')
    >>> si_two = filter_sample_info(sample_info,
    ...                             condition1=['Astral2', 'Astral5'])
    Returns a fresh DataFrame with the index reset.
    """
    mask = pd.Series(True, index=sample_info.index)
    for col, val in conditions.items():
        if col not in sample_info.columns:
            raise ValueError(
                f'column {col!r} not in sample_info '
                f'(available: {list(sample_info.columns)})'
            )
        available = sorted(sample_info[col].dropna().unique().tolist())
        if isinstance(val, (list, tuple, set)):
            missing = [v for v in val if v not in available]
            if missing:
                raise ValueError(
                    f'value(s) {missing!r} for column {col!r} not found in '
                    f'sample_info (available: {available!r})'
                )
            mask &= sample_info[col].isin(val)
        else:
            if val not in available:
                raise ValueError(
                    f'value {val!r} for column {col!r} not found in '
                    f'sample_info (available: {available!r})'
                )
            mask &= (sample_info[col] == val)
    result = sample_info[mask].reset_index(drop=True)
    if result.empty:
        raise ValueError(
            'filter_sample_info: combined filter produced an empty subset; '
            f'check the conjunction of: {dict(conditions)!r}'
        )
    return result
def filter_outlier_runs(
    df,
    sample_info,
    *,
    cohort_col=None,
    p_good_threshold=0.5,
    min_runs_for_fit=4,
    plot=True,
    figsize=None,
    palette=None,
):
    """Drop technical-outlier runs via a 2-component Gaussian Mixture on
    log10(precursor count per run). Optional, opt-in QC step.
    Per-run precursor count is the metric because precursors are the raw
    detected analytes — protein-group counts are an aggregation downstream
    that can be inflated/deflated by shared-peptide assignment. The
    threshold is the equal-posterior boundary between the "good" (higher
    mean) and "fail" cluster — no magic number to pick.
    Parameters:
      cohort_col:        sample_info column defining per-cohort fits, or
                         None for a single global fit across all runs.
                         Use a cohort column (e.g. 'condition1') whenever
                         conditions can have naturally different ID counts
                         (gradient / load / instrument / cell type) —
                         otherwise GMM mistakes biological differences for
                         technical failure. A None / global fit is right
                         for single-cohort experiments where every sample
                         should look alike.
      p_good_threshold:  posterior probability cutoff for the "good"
                         cluster (default 0.5 = equal-posterior boundary).
      min_runs_for_fit:  cohorts with fewer runs than this keep everything
                         (GMM can't fit reliably). Default 4.
      plot:              when True (default), draw one diagnostic panel per
                         cohort with the histogram, fitted Gaussians, and
                         threshold line. Always shown so the user can sanity
                         check the fit before trusting the drop.
    Returns (df_filtered, sample_info_filtered, summary). `summary` is one
    row per cohort with `n_total`, `n_kept`, `n_dropped`, `threshold`, and
    a list of `dropped_runs`.
    Both `df` and `sample_info` are returned as filtered copies — re-assign
    them in the notebook so every downstream cell sees the surviving runs:
        df, sample_info, qc = core.filter_outlier_runs(
            df, sample_info, cohort_col='condition1')
    """
    from .plotting._style import PALETTE_SINGLE, _hide_top_right_spines
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import brentq
    if cohort_col is not None and cohort_col not in sample_info.columns:
        raise ValueError(
            f'cohort_col={cohort_col!r} not in sample_info columns '
            f'({list(sample_info.columns)})'
        )
    pr_per_run = df.groupby('run', sort=False)['precursor_id'].nunique()
    if cohort_col is None:
        run_to_cohort = pd.Series(
            'all', index=sample_info['run'].astype(str), name='cohort')
    else:
        run_to_cohort = (sample_info.set_index('run')[cohort_col]
                                    .astype(str).rename('cohort'))
    cohorts = list(dict.fromkeys(run_to_cohort.loc[
        run_to_cohort.index.intersection(pr_per_run.index)
    ].values))
    if plot:
        if figsize is None:
            figsize = (max(7, 4 * len(cohorts)), 4)
        fig, axes = plt.subplots(1, len(cohorts), figsize=figsize,
                                 squeeze=False)
        axes = axes[0]
        if palette is None:
            palette = PALETTE_SINGLE
    keep_runs = []
    summary_rows = []
    for i, cohort in enumerate(cohorts):
        runs_in_cohort = [r for r in pr_per_run.index
                          if run_to_cohort.get(r) == cohort]
        vals = pr_per_run.loc[runs_in_cohort].values
        n_runs = len(vals)
        title_base = cohort if cohort != 'all' else 'all samples'
        if n_runs < min_runs_for_fit or vals.std() == 0:
            warnings.warn(
                f'filter_outlier_runs: cohort {cohort!r} has {n_runs} runs '
                f'(need >= {min_runs_for_fit}); keeping all.'
            )
            keep_runs.extend(runs_in_cohort)
            summary_rows.append({
                'cohort': cohort, 'n_total': n_runs, 'n_kept': n_runs,
                'n_dropped': 0, 'threshold': float('nan'), 'dropped_runs': [],
            })
            if plot:
                ax = axes[i]
                ax.hist(np.log10(vals + 1), bins=max(5, n_runs), alpha=0.5,
                        color=palette[i % len(palette)])
                ax.set_title(f'{title_base}\n(kept all — n={n_runs} too few to fit)',
                             fontsize=10)
                ax.set_xlabel('log10(precursor count)')
                _hide_top_right_spines(ax)
            continue
        x = np.log10(vals).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(x)
        good_label = int(np.argmax(gmm.means_.ravel()))
        p_good = gmm.predict_proba(x)[:, good_label]
        try:
            boundary = brentq(
                lambda t: gmm.predict_proba([[t]])[0, good_label] - p_good_threshold,
                x.min(), x.max(),
            )
            threshold = 10 ** boundary
        except ValueError:
            boundary = float('nan')
            threshold = float('nan')
            p_good = np.ones(n_runs)
            warnings.warn(
                f'filter_outlier_runs: cohort {cohort!r} GMM has no '
                f'equal-posterior point in range; keeping all.'
            )
        mask = p_good >= p_good_threshold
        cohort_keep = [r for r, m in zip(runs_in_cohort, mask) if m]
        cohort_drop = [r for r, m in zip(runs_in_cohort, mask) if not m]
        keep_runs.extend(cohort_keep)
        if min(gmm.weights_) < 0.02:
            warnings.warn(
                f'filter_outlier_runs: cohort {cohort!r} minority cluster '
                f'weight {min(gmm.weights_):.2%} — distribution may not be '
                f'genuinely bimodal; review the diagnostic plot.'
            )
        summary_rows.append({
            'cohort': cohort,
            'n_total': n_runs,
            'n_kept': len(cohort_keep),
            'n_dropped': len(cohort_drop),
            'threshold': threshold,
            'dropped_runs': cohort_drop,
        })
        if plot:
            ax = axes[i]
            color = palette[i % len(palette)]
            ax.hist(x.ravel(), bins=min(40, max(5, n_runs)),
                    density=True, alpha=0.45, color=color)
            xx = np.linspace(x.min() - 0.05, x.max() + 0.05, 400)
            for k in range(2):
                var = gmm.covariances_[k, 0, 0]
                pdf = gmm.weights_[k] * (
                    np.exp(-0.5 * ((xx - gmm.means_[k, 0]) / np.sqrt(var)) ** 2)
                    / np.sqrt(2 * np.pi * var)
                )
                label = 'good' if k == good_label else 'fail'
                ax.plot(xx, pdf,
                        color='black' if k == good_label else '#888',
                        linestyle='-' if k == good_label else '--',
                        label=label)
            if not np.isnan(boundary):
                ax.axvline(boundary, color='red', linestyle='--', linewidth=1.2,
                           label=f'thr ≈ {int(threshold):,}')
            ax.set_title(
                f'{title_base}\n{len(cohort_keep)}/{n_runs} kept '
                f'({len(cohort_drop)} dropped)',
                fontsize=10,
            )
            ax.set_xlabel('log10(precursor count)')
            ax.legend(fontsize=8, frameon=False)
            _hide_top_right_spines(ax)
    if plot:
        axes[0].set_ylabel('density')
        plt.tight_layout()
        plt.show()
    summary = pd.DataFrame(summary_rows)
    df_filtered = df[df['run'].isin(keep_runs)].reset_index(drop=True)
    si_filtered = sample_info[sample_info['run'].isin(keep_runs)].reset_index(drop=True)
    n_total = len(sample_info)
    n_kept = len(si_filtered)
    print(f'filter_outlier_runs: kept {n_kept}/{n_total} runs '
          f'(dropped {n_total - n_kept}) across {len(cohorts)} cohort(s).')
    for row in summary_rows:
        for r in row['dropped_runs']:
            print(f'  dropped [{row["cohort"]}] {r}: '
                  f'{int(pr_per_run.loc[r]):,} precursors')
    return df_filtered, si_filtered, summary
