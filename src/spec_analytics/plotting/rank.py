"""Rank-intensity plot.

Extracted from _core.py (REFACTOR_PLAN.md step 5); behaviour unchanged. Heavy plotting deps (matplotlib, seaborn, scipy, sklearn) are imported lazily inside the functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._style import (PALETTE_SINGLE, _hide_top_right_spines, _LEVEL_COLS, _resolve_highlights)


def plot_rank(
    df,
    sample_info=None,
    *,
    level='protein',
    group_col='condition2',
    hue_col=None,
    figsize=(7, 5),
    palette=None,
    point_size=6,
    alpha=0.5,
    highlight_genes=None,
    highlight_ids=None,
    highlight_color='#FFD700',
    highlight_size=60,
    label_highlighted=True,
    title=None,
):
    """
    Rank plot: log10(mean intensity) vs descending rank, one curve per group.

    `level` selects the entity type:
      'protein'   uses protein_group + pg_intensity
      'peptide'   uses peptide_id    + peptide_intensity
      'precursor' uses precursor_id  + precursor_intensity

    If `sample_info` is given, one rank curve is drawn per unique value of
    `sample_info[group_col]`. Pass `hue_col=` to draw one curve per
    (group_col, hue_col) pair instead — useful when the dataset spans
    e.g. multiple instruments and you want to see each curve separately.
    If `sample_info` is None, a single combined curve is drawn.

    Highlighting:
      Pass `highlight_genes=['ALB', 'GAPDH']` (case-insensitive, resolved via
      the `genes` column) or `highlight_ids=[...]` to match the entity id
      directly (`protein_group` / `peptide_id` / `precursor_id`). Highlighted
      points are drawn on top in `highlight_color` and labelled by gene
      (or id) when `label_highlighted=True`. Both highlights appear in every
      group's rank curve, since the same protein typically has different
      ranks across conditions.

    Returns `(fig, ax, rank_df)`; `rank_df` has columns `group`,
    (`hue` when hue_col is set), `rank`, the entity id column,
    `log10_mean_intensity`, `highlighted`, `label`.
    """
    import matplotlib.pyplot as plt
    if level not in _LEVEL_COLS:
        raise ValueError(f"level must be 'protein' | 'peptide' | 'precursor', got {level!r}")
    id_col, val_col = _LEVEL_COLS[level]
    palette = palette if palette is not None else PALETTE_SINGLE

    highlight_set, label_map = _resolve_highlights(
        df, id_col, highlight_genes, highlight_ids
    )

    # Build the list of (label, run_set) pairs to draw curves for.
    if sample_info is None:
        partitions = [('all', None, None)]
    elif hue_col is None:
        partitions = []
        for grp in list(sample_info[group_col].dropna().unique()):
            runs = set(sample_info.loc[sample_info[group_col] == grp, 'run'])
            partitions.append((grp, None, runs))
    else:
        partitions = []
        keys = [group_col, hue_col]
        for combo, sub in sample_info.groupby(keys, sort=False):
            grp, hue = combo if isinstance(combo, tuple) else (combo, None)
            partitions.append((grp, hue, set(sub['run'])))

    fig, ax = plt.subplots(figsize=figsize)
    long_rows = []
    for i, (grp, hue, runs) in enumerate(partitions):
        sub = df if runs is None else df[df['run'].isin(runs)]
        sub = sub.dropna(subset=[id_col, val_col])
        # Mean taken IN LOG SPACE (geometric mean), so the plotted value is the
        # location estimator on the log10 axis being drawn. log10 of a linear
        # mean sits ~sigma^2/2 higher for a log-normal, which lifts noisy
        # low-abundance entities and compresses the apparent dynamic range.
        sub = sub[sub[val_col] > 0]
        means = 10.0 ** np.log10(sub[val_col]).groupby(sub[id_col]).mean()
        means = means.sort_values(ascending=False)
        if means.empty:
            continue
        log10_means = np.log10(means.values)
        ranks = np.arange(1, len(log10_means) + 1)
        color = palette[i % len(palette)]
        if sample_info is None:
            label = f'n={len(means):,}'
        elif hue is None:
            label = f'{grp} (n={len(means):,})'
        else:
            label = f'{grp} / {hue} (n={len(means):,})'
        ax.scatter(ranks, log10_means, s=point_size, alpha=alpha,
                   c=color, edgecolor='none', label=label)

        ent_array = means.index.to_numpy()
        for r, ent, v in zip(ranks, ent_array, log10_means):
            row = {
                'group': grp,
                'rank': int(r),
                id_col: ent,
                'log10_mean_intensity': float(v),
                'highlighted': ent in highlight_set,
                'label': label_map.get(ent, ''),
            }
            if hue_col:
                row['hue'] = hue
            long_rows.append(row)

        if highlight_set:
            mask = np.array([ent in highlight_set for ent in ent_array])
            if mask.any():
                hi_ranks = ranks[mask]
                hi_vals = log10_means[mask]
                hi_ents = ent_array[mask]
                ax.scatter(hi_ranks, hi_vals,
                           s=highlight_size, c=highlight_color,
                           edgecolor='black', linewidth=0.6, alpha=1.0,
                           zorder=10)
                if label_highlighted:
                    for r, v, ent in zip(hi_ranks, hi_vals, hi_ents):
                        ax.annotate(label_map.get(ent, str(ent)),
                                    (r, v), fontsize=8, fontweight='bold',
                                    xytext=(5, 5), textcoords='offset points',
                                    zorder=11)

    ax.set_xlabel(f'{level.capitalize()} rank (most -> least abundant)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'log₁₀ mean {level} intensity', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='upper right')
    if title is None:
        title = f'{level.capitalize()} rank plot'
    ax.set_title(title, fontsize=13, fontweight='bold')
    _hide_top_right_spines(ax)
    plt.tight_layout()

    return fig, ax, pd.DataFrame(long_rows)
