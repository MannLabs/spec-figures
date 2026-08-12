"""Distribution histograms: intensity, peptide GRAVY, peptide length, and sequence coverage.

Extracted from _core.py (REFACTOR_PLAN.md step 5); behaviour unchanged. Heavy plotting deps (matplotlib, seaborn, scipy, sklearn) are imported lazily inside the functions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._style import (PALETTE_SINGLE, _empty_plot_with_message, _hide_top_right_spines, _LEVEL_COLS)
from ..sequences import (digest_protein, gravy, compute_theoretical_coverage, compute_protein_info)


def plot_intensity_histogram(
    df,
    sample_info,
    *,
    level='precursor',
    group_col='condition2',
    hue_col=None,
    figsize=(6, 4),
    palette=None,
    alpha=0.6,
    bins=50,
    edgecolor='black',
    edge_linewidth=0.5,
    title=None,
):
    """
    Overlapping histograms of log2 mean intensity per group.

    For each value of `sample_info[group_col]`, computes the mean intensity
    of each entity across the runs in that group, log2-transforms, and plots
    the distribution.

    Pass `hue_col=` to draw one histogram per (group, hue) cross product —
    useful for multi-instrument experiments. (group, hue) cells with no
    measurements are skipped.

    Returns `(fig, ax, histogram_data)` where `histogram_data` is a long
    DataFrame with columns `[group, (hue,) id, log2_mean_intensity]`.
    """
    import matplotlib.pyplot as plt
    if level not in _LEVEL_COLS:
        raise ValueError(f"level must be 'protein' | 'peptide' | 'precursor', got {level!r}")
    id_col, val_col = _LEVEL_COLS[level]
    palette = palette if palette is not None else PALETTE_SINGLE

    if hue_col is None:
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

    by_partition = {}
    all_log2 = []
    for label, hue, runs in partitions:
        sub = df[df['run'].isin(runs)].dropna(subset=[val_col])
        # Mean taken IN LOG SPACE (geometric mean) so the histogram describes
        # the distribution on the axis it is drawn on; log2 of a linear mean is
        # biased upward by ~sigma^2/2 for a log-normal, which shifts the noisy
        # low-intensity tail and narrows the quoted dynamic range.
        sub = sub[sub[val_col] > 0]
        means = 2.0 ** np.log2(sub[val_col]).groupby(sub[id_col]).mean()
        log2 = np.log2(means.values)
        if log2.size == 0:
            continue
        by_partition[(label, hue)] = (means, log2)
        all_log2.append(log2)
    if not all_log2:
        warnings.warn(
            f'plot_intensity_histogram: no positive {level} intensities in '
            f'any group — rendering an empty plot.'
        )
        fig, ax = _empty_plot_with_message(
            f'no positive {level} intensities',
            figsize=figsize, title=title)
        return fig, ax, pd.DataFrame([])
    flat = np.concatenate(all_log2)
    bin_edges = np.linspace(flat.min(), flat.max(), bins + 1)

    fig, ax = plt.subplots(figsize=figsize)
    long_rows = []
    for i, ((grp, hue), (means, log2)) in enumerate(by_partition.items()):
        color = palette[i % len(palette)]
        if hue is None:
            label = f'{grp} (n={len(log2):,})'
        else:
            label = f'{grp} / {hue} (n={len(log2):,})'
        ax.hist(log2, bins=bin_edges, alpha=alpha, color=color,
                label=label, edgecolor=edgecolor, linewidth=edge_linewidth)
        for entity_id, val in zip(means.index, log2):
            row = {'group': grp, id_col: entity_id,
                   'log2_mean_intensity': float(val)}
            if hue_col:
                row['hue'] = hue
            long_rows.append(row)

    ax.set_xlabel(f'log₂ mean {level} intensity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='upper right')
    if title is None:
        title = f'{level.capitalize()} intensity distribution by {group_col}'
        if hue_col:
            title += f' / {hue_col}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    _hide_top_right_spines(ax)
    plt.tight_layout()

    return fig, ax, pd.DataFrame(long_rows)


def plot_peptide_gravy_distribution(
    df,
    sample_info,
    *,
    group_col='condition2',
    hue_col=None,
    sequence_col='sequence',
    figsize=(6, 4),
    palette=None,
    alpha=0.6,
    bins=50,
    edgecolor='black',
    edge_linewidth=0.5,
    median_line=True,
    title=None,
):
    """Histogram of GRAVY (Kyte-Doolittle hydropathy) at peptide level.

    Pass `hue_col=` to draw one distribution per (group, hue) cross product.
    Each peptide contributes once per partition, regardless of how many runs
    or charge states detected it.

    Returns `(fig, ax, plot_df)` with one row per (group, [hue,] peptide,
    gravy) for source-data export.
    """
    import matplotlib.pyplot as plt
    palette = palette if palette is not None else PALETTE_SINGLE

    if hue_col is None:
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

    by_partition = {}
    all_vals = []
    long_rows = []
    for grp, hue, runs in partitions:
        sub = df[df['run'].isin(runs)]
        seqs = sub[sequence_col].dropna().astype(str).unique()
        gv = np.array([gravy(s) for s in seqs], dtype=float)
        mask = ~np.isnan(gv)
        seqs = seqs[mask]
        gv = gv[mask]
        if gv.size == 0:
            continue
        by_partition[(grp, hue)] = (seqs, gv)
        all_vals.append(gv)
        for s, v in zip(seqs, gv):
            row = {'group': grp, 'sequence': s, 'gravy': float(v)}
            if hue_col:
                row['hue'] = hue
            long_rows.append(row)
    flat = np.concatenate(all_vals) if all_vals else np.array([])
    if flat.size == 0:
        warnings.warn(
            'plot_peptide_gravy_distribution: no peptides with valid GRAVY '
            'in any partition — rendering an empty plot.'
        )
        fig, ax = _empty_plot_with_message(
            'no peptides with valid GRAVY',
            figsize=figsize, title=title)
        return fig, ax, pd.DataFrame(long_rows)
    bin_edges = np.linspace(flat.min(), flat.max(), bins + 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, ((grp, hue), (seqs, gv)) in enumerate(by_partition.items()):
        color = palette[i % len(palette)]
        if hue is None:
            label = f'{grp} (n={len(gv):,})'
        else:
            label = f'{grp} / {hue} (n={len(gv):,})'
        ax.hist(gv, bins=bin_edges, alpha=alpha, color=color, label=label,
                edgecolor=edgecolor, linewidth=edge_linewidth)
        if median_line and len(gv):
            ax.axvline(float(np.median(gv)), color=color,
                       linestyle='--', linewidth=1.5)

    ax.axvline(0, color='black', linewidth=0.6, alpha=0.4, zorder=0)
    ax.set_xlabel('GRAVY (Kyte-Doolittle)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peptides', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, loc='upper right')
    if title is None:
        title = f'Peptide GRAVY by {group_col}'
        if hue_col:
            title += f' / {hue_col}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax, pd.DataFrame(long_rows)


def plot_peptide_length_distribution(
    df,
    sample_info,
    *,
    group_col='condition2',
    hue_col=None,
    sequence_col='sequence',
    show_theoretical=False,
    protein_sequences=None,
    protease='trypsin',
    max_missed_cleavages=1,
    keil_rule=False,
    min_peptide_length=6,
    max_peptide_length=40,
    palette=None,
    theoretical_color='#888888',
    theoretical_label='theoretical (perfect digest)',
    figsize=(7, 4),
    alpha=0.6,
    bin_width=1,
    edgecolor='black',
    edge_linewidth=0.5,
    median_line=True,
    mode='density',
    title=None,
):
    """Histogram of observed peptide sequence lengths per condition.

    Per-condition: unique observed sequences from `df[sequence_col]` filtered
    to runs in each `sample_info[group_col]` value. Each peptide counted once.

    `show_theoretical=True` (with `protein_sequences`) overlays a single grey
    histogram of in-silico tryptic-peptide lengths obtained by digesting every
    detected protein_group leader exactly once. The digest parameters
    (`protease`, `max_missed_cleavages`, `keil_rule`, `min_peptide_length`,
    `max_peptide_length`) match `compute_theoretical_coverage` so the same
    settings dict can be unpacked into both calls.

    `mode='density'` (default) plots probability density so distributions of
    very different sizes — typically the theoretical set is much larger than
    any single observed condition — are visually comparable on the same
    axis. Pass `mode='count'` for raw counts, which makes the size mismatch
    with the theoretical distribution visible.

    Returns `(fig, ax, plot_df)` with one row per (group, peptide, length).
    """
    if mode not in ('density', 'count'):
        raise ValueError(
            f"mode must be 'density' or 'count', got {mode!r}"
        )
    density = (mode == 'density')
    import matplotlib.pyplot as plt
    palette = palette if palette is not None else PALETTE_SINGLE

    if hue_col is None:
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

    by_partition = {}
    all_lengths = []
    long_rows = []
    for grp, hue, runs in partitions:
        sub = df[df['run'].isin(runs)]
        seqs = sub[sequence_col].dropna().astype(str).unique()
        seqs = np.array([s for s in seqs if s], dtype=object)
        if seqs.size == 0:
            continue
        lens = np.array([len(s) for s in seqs], dtype=int)
        by_partition[(grp, hue)] = (seqs, lens)
        all_lengths.append(lens)
        for s, L in zip(seqs, lens):
            row = {'group': grp, 'sequence': s, 'length': int(L)}
            if hue_col:
                row['hue'] = hue
            long_rows.append(row)

    th_lengths = None
    if show_theoretical:
        if protein_sequences is None:
            raise ValueError(
                'show_theoretical=True requires protein_sequences='
            )
        # Digest each leader exactly once across the full set of detected
        # protein groups.
        leaders = set()
        for pg in df['protein_group'].dropna().astype(str).unique():
            leaders.add(pg.split(';')[0])
        th_buf = []
        for leader in leaders:
            seq = protein_sequences.get(leader)
            if not seq:
                continue
            for start, end, _pep in digest_protein(
                seq, protease=protease,
                max_missed_cleavages=max_missed_cleavages,
                keil_rule=keil_rule,
            ):
                L = end - start
                if L < min_peptide_length:
                    continue
                if max_peptide_length is not None and L > max_peptide_length:
                    continue
                th_buf.append(L)
        th_lengths = np.array(th_buf, dtype=int) if th_buf else np.array([], dtype=int)
        if th_lengths.size:
            all_lengths.append(th_lengths)

    flat = np.concatenate(all_lengths) if all_lengths else np.array([], dtype=int)
    if flat.size == 0:
        warnings.warn(
            'plot_peptide_length_distribution: no peptides found across '
            'groups — rendering an empty plot.'
        )
        fig, ax = _empty_plot_with_message(
            'no peptides found',
            figsize=figsize, title=title)
        return fig, ax, pd.DataFrame(long_rows)
    lo = int(flat.min())
    hi = int(flat.max())
    bin_edges = np.arange(lo, hi + bin_width + 1, bin_width)

    fig, ax = plt.subplots(figsize=figsize)
    th_median = None
    if th_lengths is not None and th_lengths.size:
        ax.hist(th_lengths, bins=bin_edges, alpha=alpha, color=theoretical_color,
                label=f'{theoretical_label} (n={len(th_lengths):,})',
                edgecolor=edgecolor, linewidth=edge_linewidth, zorder=1,
                density=density)
        th_median = float(np.median(th_lengths))
        if median_line:
            ax.axvline(th_median, color=theoretical_color,
                       linestyle='--', linewidth=1.5, zorder=2)

    for i, ((grp, hue), (_, lens)) in enumerate(by_partition.items()):
        color = palette[i % len(palette)]
        if hue is None:
            label = f'{grp} (n={len(lens):,})'
        else:
            label = f'{grp} / {hue} (n={len(lens):,})'
        ax.hist(lens, bins=bin_edges, alpha=alpha, color=color, label=label,
                edgecolor=edgecolor, linewidth=edge_linewidth, zorder=3,
                density=density)
        if median_line and len(lens):
            ax.axvline(float(np.median(lens)), color=color,
                       linestyle='--', linewidth=1.5, zorder=4)

    ax.set_xlabel('Peptide length (AA)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density' if mode == 'density' else 'Peptides',
                  fontsize=12, fontweight='bold')
    legend_loc = 'upper right' if th_median is None else 'upper center'
    ax.legend(frameon=False, loc=legend_loc)
    if title is None:
        title = f'Peptide length by {group_col}'
        if hue_col:
            title += f' / {hue_col}'
        if th_median is not None:
            title += ' (observed vs theoretical)'
    ax.set_title(title, fontsize=13, fontweight='bold')
    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax, pd.DataFrame(long_rows)


def plot_coverage_histogram(
    protein_info,
    *,
    group_col='group',
    hue_col=None,
    value_col='coverage_pct',
    theoretical_col=None,
    theoretical_color='#888888',
    theoretical_label='theoretical',
    figsize=(6, 4),
    palette=None,
    alpha=0.6,
    bins=50,
    edgecolor='black',
    edge_linewidth=0.5,
    median_line=True,
    median_decimals=1,
    title=None,
):
    """Overlapping histograms of sequence coverage with median dashed lines.

    Operates on the output of `compute_protein_info` (or any DataFrame with
    `group_col` + `value_col`). One histogram per unique `group_col` value;
    pass `hue_col=` to draw one curve per (group, hue) instead — typically
    used after `compute_protein_info(..., hue_col=...)`.

    Pass `theoretical_col='theoretical_coverage_pct'` to overlay a single
    grey histogram of the theoretical maximum coverage — deduplicated per
    protein_group, so it represents the in-silico baseline rather than a
    per-condition curve.

    Returns `(fig, ax, plot_df)` where `plot_df` is the rows actually plotted
    (NaN-filtered) for source-data export.
    """
    import matplotlib.pyplot as plt
    palette = palette if palette is not None else PALETTE_SINGLE
    sub = protein_info.dropna(subset=[value_col]).copy()
    if sub.empty:
        warnings.warn(
            f'plot_coverage_histogram: no rows with finite {value_col!r} '
            f'— rendering an empty plot.'
        )
        out_cols = ['protein_group', group_col, value_col]
        if hue_col:
            out_cols.insert(2, hue_col)
        fig, ax = _empty_plot_with_message(
            f'no rows with finite {value_col}',
            figsize=figsize, title=title)
        return fig, ax, sub[out_cols].copy()

    if hue_col is None:
        partitions = []
        for grp in list(sub[group_col].dropna().unique()):
            partitions.append(((grp, None),
                               sub.loc[sub[group_col] == grp, value_col].to_numpy()))
    else:
        partitions = []
        keys = [group_col, hue_col]
        for combo, sub_part in sub.groupby(keys, sort=False):
            grp, hue = combo if isinstance(combo, tuple) else (combo, None)
            partitions.append(((grp, hue), sub_part[value_col].to_numpy()))

    fig, ax = plt.subplots(figsize=figsize)
    vmin = float(sub[value_col].min())
    vmax = float(sub[value_col].max())

    # Pull theoretical values up front so the bin edges can span both ranges.
    th_vals = None
    if theoretical_col is not None and theoretical_col in protein_info.columns:
        th_vals = (protein_info.dropna(subset=[theoretical_col])
                              .drop_duplicates('protein_group')[theoretical_col]
                              .to_numpy())
        if len(th_vals):
            vmin = min(vmin, float(np.min(th_vals)))
            vmax = max(vmax, float(np.max(th_vals)))
    bin_edges = np.linspace(vmin, vmax, bins + 1)

    # Theoretical drawn first so observed sits visually on top.
    th_median = None
    if th_vals is not None and len(th_vals):
        ax.hist(th_vals, bins=bin_edges, alpha=alpha, color=theoretical_color,
                label=f'{theoretical_label} (n={len(th_vals):,})',
                edgecolor=edgecolor, linewidth=edge_linewidth, zorder=1)
        th_median = float(np.median(th_vals))
        if median_line:
            ax.axvline(th_median, color=theoretical_color,
                       linestyle='--', linewidth=1.5, zorder=2)

    partition_medians = []
    for i, ((grp, hue), vals) in enumerate(partitions):
        if vals.size == 0:
            partition_medians.append(((grp, hue), float('nan')))
            continue
        color = palette[i % len(palette)]
        if hue is None:
            label = f'{grp} (n={len(vals):,})'
        else:
            label = f'{grp} / {hue} (n={len(vals):,})'
        ax.hist(vals, bins=bin_edges, alpha=alpha, color=color, label=label,
                edgecolor=edgecolor, linewidth=edge_linewidth, zorder=3)
        med = float(np.median(vals))
        partition_medians.append(((grp, hue), med))
        if median_line and not np.isnan(med):
            ax.axvline(med, color=color, linestyle='--', linewidth=1.5, zorder=4)

    if median_line:
        ymax = ax.get_ylim()[1]
        labelled_idx = [k for k, (_, m) in enumerate(partition_medians)
                        if not np.isnan(m)]
        if th_median is not None:
            labelled_idx = labelled_idx + ['__theoretical__']
        # Sort by median value so left-most label sits highest.
        def _med_for(k):
            if k == '__theoretical__':
                return th_median
            return partition_medians[k][1]
        order = sorted(labelled_idx, key=_med_for)
        height_for = {k: 0.97 - 0.08 * t for t, k in enumerate(order)}
        for k, ((grp, hue), med) in enumerate(partition_medians):
            if k not in height_for or np.isnan(med):
                continue
            color = palette[k % len(palette)]
            ax.text(med, ymax * height_for[k], f'{med:.{median_decimals}f}%',
                    color=color, ha='center', va='top',
                    fontsize=11, fontweight='bold')
        if th_median is not None:
            ax.text(th_median, ymax * height_for['__theoretical__'],
                    f'{th_median:.{median_decimals}f}%',
                    color=theoretical_color, ha='center', va='top',
                    fontsize=11, fontweight='bold')

    ax.set_xlabel('Sequence coverage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Protein groups', fontsize=12, fontweight='bold')
    legend_loc = 'upper center' if th_median is not None else 'upper right'
    ax.legend(frameon=False, loc=legend_loc)
    if title is None:
        title = f'Sequence coverage by {group_col}'
        if hue_col:
            title += f' / {hue_col}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    _hide_top_right_spines(ax)
    plt.tight_layout()
    out_cols = ['protein_group', group_col, value_col]
    if hue_col:
        out_cols.insert(2, hue_col)
    return fig, ax, sub[out_cols].copy()
