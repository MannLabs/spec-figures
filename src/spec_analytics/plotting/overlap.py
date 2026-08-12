"""Set-overlap plots: Venn, shared/unique bars, UpSet.

Extracted from _core.py (REFACTOR_PLAN.md step 5); behaviour unchanged. Heavy plotting deps (matplotlib, seaborn, scipy, sklearn) are imported lazily inside the functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._style import (PALETTE_SINGLE, _hide_top_right_spines, _LEVEL_COLS,
                     _resolve_panel_size)
from ..reshape import add_combined_group


def plot_completeness(
    df, sample_info, *, level='precursor', group_col='condition2',
    group_order=None, palette=None, normalize=False, figsize=None,
    title=None, y_label=None, label_fontsize=10, tick_fontsize=10,
    legend_fontsize=8, verbose=True, ax=None,
):
    """Identifications retained against a "detected in >= k of N replicates" cut.

    The counterweight to an identification bar chart. A method can win on total
    count while its extra entities are sporadic — seen in one replicate of four
    and missing from the rest — and this curve is where that shows, as a steeper
    fall from k=1 to k=N.

    Two orientations:

    ``normalize=False`` (default)
        absolute entities surviving each threshold. The k=1 point IS the bar
        chart's number, so the two panels tie together.
    ``normalize=True``
        the same divided by each condition's own k=1 count, so every curve
        starts at 1.0 and only the *shape* is compared. This is what separates
        reproducibility from depth.

    Oriented as **retention** (falling from k=1) rather than as a cumulative
    "detected in <= k" distribution. The textbook sigmoid only appears when
    completeness is poor; with good DIA data most entities sit at k=N, so a CDF
    collapses into a step at the right edge and reads as a hockey stick, while
    the retention curve stays legible and is what actually gets quoted ("59% of
    precursors survive a full-completeness filter").

    One curve per condition, one axes per level. The twin-axis 1x2 form suits a
    single condition at two levels; with several conditions, mixing levels on
    twinned axes puts six curves in one panel.

    :returns: ``(fig, ax, source_df)`` with every plotted point.
    """
    import matplotlib.pyplot as plt

    id_col, val_col = _LEVEL_COLS[level]
    palette = palette if palette is not None else PALETTE_SINGLE

    groups = (list(group_order) if group_order is not None
              else list(sample_info[group_col].dropna().unique()))
    if isinstance(palette, (list, tuple)):
        palette = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    work = df.dropna(subset=[id_col, val_col])
    work = work[work[val_col] > 0]

    rows, max_n = [], 1
    for group in groups:
        runs = list(sample_info.loc[sample_info[group_col] == group, 'run'])
        if not runs:
            continue
        max_n = max(max_n, len(runs))
        # How many of this condition's runs each entity was quantified in.
        seen = (work[work['run'].isin(runs)]
                .drop_duplicates([id_col, 'run'])
                .groupby(id_col).size())
        ks = np.arange(1, len(runs) + 1)
        counts = np.array([int((seen >= k).sum()) for k in ks])
        union = counts[0] if counts.size and counts[0] > 0 else 1
        ax.plot(ks, counts / union if normalize else counts, marker='o',
                markersize=4, lw=1.6, color=palette[group], label=str(group))
        rows.extend({'condition': group, 'min_replicates': int(k),
                     'n_retained': int(c),
                     'fraction_retained': float(c / union)}
                    for k, c in zip(ks, counts))
        if verbose:
            print(f'    {group}: {counts[0]:,} at k=1 -> {counts[-1]:,} at '
                  f'k={len(runs)} ({counts[-1] / union:.1%} retained)')

    ax.set_xticks(np.arange(1, max_n + 1))
    ax.set_xlabel(f'Detected in ≥ k of {max_n} replicates',
                  fontsize=label_fontsize)
    if y_label is None:
        y_label = ('Fraction retained' if normalize else
                   f'# {"protein groups" if level == "protein" else level + "s"}')
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_ylim(0, 1.02 if normalize else None)
    if not normalize:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(frameon=False, fontsize=legend_fontsize, loc='lower left')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_venn(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    figsize=(6, 5),
    palette=None,
    title=None,
):
    """Venn-style overlap of detected entities per `sample_info[group_col]`.

    Dispatches by group count:
      2-3 groups: matplotlib_venn (proportional ellipses).
      4-6 groups: the `venn` package (Edwards-style fixed-position ellipses).
      >6 groups: not drawable as a Venn — raises with a suggestion to use
                 `core.add_combined_group(...)` to collapse axes.

    `level` selects what is counted:
      'protein'   uses protein_group   (default)
      'peptide'   uses peptide_id      (peptidoform-resolved)
      'precursor' uses precursor_id

    Each group's entity set is the union over its runs of all non-null ids
    at that level. Returns `(fig, ax, source_df)` where `source_df` is a
    long table `[group, id]` for source-data export.
    """
    import matplotlib.pyplot as plt
    if level not in _LEVEL_COLS:
        raise ValueError(f"level must be 'protein' | 'peptide' | 'precursor', got {level!r}")
    id_col, _ = _LEVEL_COLS[level]
    palette = palette if palette is not None else PALETTE_SINGLE

    groups = list(sample_info[group_col].dropna().unique())
    if len(groups) < 2:
        raise ValueError(
            f'plot_venn needs >= 2 groups in {group_col!r}, '
            f'got {len(groups)}: {groups}'
        )
    if len(groups) > 6:
        raise ValueError(
            f'plot_venn cannot draw a Venn for {len(groups)} groups '
            f'(supports up to 6). Collapse axes with '
            f'`core.add_combined_group(...)` or compare subsets one at a time.'
        )

    sets = {}
    rows = []
    for grp in groups:
        runs = set(sample_info.loc[sample_info[group_col] == grp, 'run'])
        ids = df.loc[df['run'].isin(runs), id_col].dropna().unique()
        sets[grp] = set(ids)
        for ent in sets[grp]:
            rows.append({'group': grp, id_col: ent})

    fig, ax = plt.subplots(figsize=figsize)
    if len(groups) in (2, 3):
        # matplotlib_venn — proportional circles.
        try:
            from matplotlib_venn import venn2, venn3
        except ImportError as e:
            raise ImportError(
                'plot_venn requires matplotlib_venn for 2- and 3-way Venns; '
                'install with `pip install matplotlib_venn`'
            ) from e
        venn_fn = venn2 if len(groups) == 2 else venn3
        venn_fn([sets[g] for g in groups],
                set_labels=[str(g) for g in groups],
                ax=ax,
                set_colors=palette[:len(groups)])
    else:
        # `venn` package — Edwards-style for 4-6 sets.
        try:
            import venn as _venn
        except ImportError as e:
            raise ImportError(
                'plot_venn requires the `venn` package for 4-6-way Venns; '
                'install with `pip install venn`'
            ) from e
        data = {str(g): sets[g] for g in groups}
        # Build (r, g, b, a) colours from the palette so `venn` renders them
        # with consistent transparency.
        import matplotlib.colors as mcolors
        colors = [(*mcolors.to_rgb(palette[i % len(palette)]), 0.45)
                  for i in range(len(groups))]
        _venn.venn(data, ax=ax, cmap=colors, fontsize=11, legend_loc='upper right')

    if title is None:
        title = f'{level.capitalize()} overlap by {group_col}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_set_overlap(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    show_common=False,
    figsize=(7, 5),
    palette=None,
    shared_color='#cccccc',
    common_color='#888888',
    bar_width=0.6,
    show_values=True,
    label_fontsize=11,
    value_fontsize=None,
    tick_fontsize=10,
    legend_fontsize=10,
    title=None,
):
    """Stacked-bar 'unique vs shared' summary across `sample_info[group_col]`.

    Cleaner than a Venn diagram when you have many conditions (4+). For each
    group, partitions the entities detected in its runs into:

      'unique'  — found only in this group.
      'shared'  — found in this group and at least one other group (but not
                  necessarily all). Only shown if `show_common=False` (default)
                  this segment counts entities in 2..N groups; with
                  `show_common=True` it counts entities in 2..N-1 groups only.
      'common'  — found in EVERY group. Only included when `show_common=True`.

    `value_fontsize` sizes the in-bar segment counts (and, +1, the per-group
    total above each bar) independently of the axis `label_fontsize`; it
    defaults to `label_fontsize - 1`. Worth turning down on a narrow panel —
    the in-segment counts are drawn white-on-fill, so a label wider than the
    bar spills onto the white background and becomes unreadable.

    Bar colours: 'unique' uses the group's palette colour. 'shared' is
    `shared_color` (light grey by default). 'common' is `common_color`
    (medium grey).

    Returns `(fig, ax, source_df)` where `source_df` has one row per group
    with `unique`, `shared`, and (if requested) `common` counts.
    """
    import matplotlib.pyplot as plt
    if level not in _LEVEL_COLS:
        raise ValueError(f"level must be 'protein' | 'peptide' | 'precursor', got {level!r}")
    id_col, _ = _LEVEL_COLS[level]
    palette = palette if palette is not None else PALETTE_SINGLE

    value_fontsize = label_fontsize - 1 if value_fontsize is None else value_fontsize

    groups = list(sample_info[group_col].dropna().unique())
    if len(groups) < 2:
        raise ValueError(
            f'plot_set_overlap needs >= 2 groups in {group_col!r}, '
            f'got {len(groups)}: {groups}'
        )

    sets = {}
    for grp in groups:
        runs = set(sample_info.loc[sample_info[group_col] == grp, 'run'])
        sets[grp] = set(df.loc[df['run'].isin(runs), id_col].dropna().unique())

    common = set.intersection(*sets.values()) if show_common else set()

    rows = []
    for grp in groups:
        others = set.union(*[sets[g] for g in groups if g != grp])
        unique = sets[grp] - others
        if show_common:
            shared = sets[grp] - unique - common
            rows.append({'group': grp, 'unique': len(unique),
                         'shared': len(shared), 'common': len(common)})
        else:
            shared = sets[grp] - unique
            rows.append({'group': grp, 'unique': len(unique),
                         'shared': len(shared)})

    summary = pd.DataFrame(rows)
    xs = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=figsize)
    # 'unique' bottom in the group's primary colour.
    unique_vals = summary['unique'].to_numpy()
    unique_colors = [palette[i % len(palette)] for i in range(len(groups))]
    bars_u = ax.bar(xs, unique_vals, bar_width, color=unique_colors,
                    edgecolor='black', linewidth=0.5, label='unique')
    bottom = unique_vals.copy()

    shared_vals = summary['shared'].to_numpy()
    bars_s = ax.bar(xs, shared_vals, bar_width, bottom=bottom,
                    color=shared_color, edgecolor='black', linewidth=0.5,
                    label='shared')
    bottom = bottom + shared_vals

    if show_common:
        common_vals = summary['common'].to_numpy()
        bars_c = ax.bar(xs, common_vals, bar_width, bottom=bottom,
                        color=common_color, edgecolor='black', linewidth=0.5,
                        label='common (all)')
        bottom = bottom + common_vals

    if show_values:
        # Per-segment counts in the centre of each segment.
        def _annotate(bars, base):
            for bar, val, b in zip(bars, base, bottom * 0):
                pass  # noop; we use explicit loop below
        for i, b in enumerate(bars_u):
            v = unique_vals[i]
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_y() + b.get_height() / 2,
                        f'{int(v):,}', ha='center', va='center',
                        fontsize=value_fontsize, fontweight='bold',
                        color='white')
        for i, b in enumerate(bars_s):
            v = shared_vals[i]
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_y() + b.get_height() / 2,
                        f'{int(v):,}', ha='center', va='center',
                        fontsize=value_fontsize, fontweight='bold',
                        color='black')
        if show_common:
            for i, b in enumerate(bars_c):
                v = common_vals[i]
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_y() + b.get_height() / 2,
                            f'{int(v):,}', ha='center', va='center',
                            fontsize=value_fontsize, fontweight='bold',
                            color='white')
        # Per-group total above each bar.
        totals = unique_vals + shared_vals + (common_vals if show_common else 0)
        for i, total in enumerate(totals):
            ax.text(xs[i], total, f'{int(total):,}', ha='center',
                    va='bottom', fontsize=value_fontsize + 1, fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_ylabel(f'{level.capitalize()} count',
                  fontsize=label_fontsize, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.legend(fontsize=legend_fontsize, frameon=False,
              loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    if title is None:
        title = f'{level.capitalize()} unique vs shared by {group_col}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax, summary


def plot_upset(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    max_intersections=20,
    min_count=1,
    figsize=None,
    palette=None,
    dot_color='black',
    absent_dot_color='#d0d0d0',
    show_set_sizes=True,
    title=None,
):
    """UpSet plot — bar chart of intersection sizes with set-membership dots.

    Cleaner than a Venn for many sets (especially >4). Each column corresponds
    to a unique set-membership signature (which subset of groups contains the
    entity); bar height is the number of entities in exactly that intersection.
    The dot matrix below shows which groups are members of each intersection
    (filled dot = in the set).

    Parameters:
      level: 'protein' | 'peptide' | 'precursor' — what is counted.
      max_intersections: cap the number of bars shown (default 20). Top-N by
        count.
      min_count: drop intersections with fewer than this many entities.
      show_set_sizes: draw a horizontal bar chart on the left with each
        group's total entity count.

    Returns (fig, axes, source_df) where `axes` is a dict
    `{'bars', 'dots', 'set_sizes'}` for downstream tweaking, and `source_df`
    has one row per intersection with columns
    `[intersection (frozenset), count]`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    if level not in _LEVEL_COLS:
        raise ValueError(f"level must be 'protein' | 'peptide' | 'precursor', got {level!r}")
    id_col, _ = _LEVEL_COLS[level]
    palette = palette if palette is not None else PALETTE_SINGLE

    groups = list(sample_info[group_col].dropna().unique())
    if len(groups) < 2:
        raise ValueError(
            f'plot_upset needs >= 2 groups in {group_col!r}, got {len(groups)}'
        )

    # Build per-group sets and the per-entity membership signature.
    sets = {}
    for grp in groups:
        runs = set(sample_info.loc[sample_info[group_col] == grp, 'run'])
        sets[grp] = set(df.loc[df['run'].isin(runs), id_col].dropna().unique())

    # signature: frozenset of groups containing the entity.
    from collections import Counter
    all_entities = set().union(*sets.values())
    signatures = Counter()
    for ent in all_entities:
        sig = frozenset(g for g in groups if ent in sets[g])
        if sig:
            signatures[sig] += 1

    # Filter + sort intersections by count (desc).
    intersections = [(sig, n) for sig, n in signatures.items() if n >= min_count]
    intersections.sort(key=lambda x: -x[1])
    intersections = intersections[:max_intersections]
    if not intersections:
        raise ValueError('no non-empty intersections found')

    n_cols = len(intersections)
    n_rows = len(groups)

    if figsize is None:
        figsize = (max(7, 0.55 * n_cols + (2.0 if show_set_sizes else 0.5)),
                   max(5, 0.6 * n_rows + 2.5))

    fig = plt.figure(figsize=figsize)
    if show_set_sizes:
        # Left column for set sizes, right columns for bars + dots.
        gs = GridSpec(2, 2, width_ratios=[1.2, 4],
                      height_ratios=[3, max(1.2, 0.4 * n_rows)],
                      hspace=0.05, wspace=0.25)
        ax_sets = fig.add_subplot(gs[1, 0])
        ax_bars = fig.add_subplot(gs[0, 1])
        ax_dots = fig.add_subplot(gs[1, 1], sharex=ax_bars)
    else:
        gs = GridSpec(2, 1,
                      height_ratios=[3, max(1.2, 0.4 * n_rows)],
                      hspace=0.05)
        ax_sets = None
        ax_bars = fig.add_subplot(gs[0, 0])
        ax_dots = fig.add_subplot(gs[1, 0], sharex=ax_bars)

    xs = np.arange(n_cols)

    # Top: intersection counts.
    counts = np.array([n for _, n in intersections])
    ax_bars.bar(xs, counts, width=0.7, color=palette[0],
                edgecolor='black', linewidth=0.5)
    for i, c in enumerate(counts):
        ax_bars.text(xs[i], c, f'{int(c):,}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')
    ax_bars.set_ylabel(f'{level.capitalize()}s in intersection',
                       fontsize=11, fontweight='bold')
    ax_bars.set_xticks([])
    ax_bars.set_xlim(-0.6, n_cols - 0.4)
    ax_bars.set_ylim(top=counts.max() * 1.18)
    _hide_top_right_spines(ax_bars)
    ax_bars.spines['bottom'].set_visible(False)

    # Bottom: dot matrix. Rows = groups (top->bottom in group order),
    # cols = intersections (left->right by descending count).
    y_positions = np.arange(n_rows)[::-1]  # top row first
    for ci, (sig, _) in enumerate(intersections):
        for ri, grp in enumerate(groups):
            y = y_positions[ri]
            colour = dot_color if grp in sig else absent_dot_color
            ax_dots.scatter(ci, y, s=110, c=colour,
                            edgecolor='black', linewidth=0.4, zorder=3)
        # Connecting line for sets in this intersection.
        members_y = [y_positions[i] for i, g in enumerate(groups) if g in sig]
        if len(members_y) >= 2:
            ax_dots.plot([ci, ci], [min(members_y), max(members_y)],
                         color=dot_color, linewidth=2.0, zorder=2)
    ax_dots.set_yticks(y_positions)
    ax_dots.set_yticklabels(groups, fontsize=10)
    ax_dots.set_xticks([])
    ax_dots.set_xlim(-0.6, n_cols - 0.4)
    ax_dots.set_ylim(-0.6, n_rows - 0.4)
    for spine in ax_dots.spines.values():
        spine.set_visible(False)
    ax_dots.tick_params(axis='both', length=0)

    # Left: per-group total set sizes.
    if ax_sets is not None:
        sizes = [len(sets[g]) for g in groups]
        ax_sets.barh(y_positions, sizes, height=0.6,
                     color=palette[0], edgecolor='black', linewidth=0.5)
        for ri, sz in enumerate(sizes):
            ax_sets.text(sz, y_positions[ri], f' {int(sz):,}',
                         va='center', ha='left', fontsize=9)
        ax_sets.invert_xaxis()
        ax_sets.set_yticks([])
        ax_sets.set_ylim(ax_dots.get_ylim())
        ax_sets.set_xlabel('Set size', fontsize=10, fontweight='bold')
        ax_sets.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
        _hide_top_right_spines(ax_sets)
        # No top spine (we removed it via the helper) but we want the bottom
        # and left visible — they already are.

    if title is None:
        title = f'{level.capitalize()} intersections by {group_col}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    source_df = pd.DataFrame({
        'intersection': [sorted(sig) for sig, _ in intersections],
        'count': counts,
        'n_groups': [len(sig) for sig, _ in intersections],
    })

    axes_dict = {'bars': ax_bars, 'dots': ax_dots, 'set_sizes': ax_sets}
    return fig, axes_dict, source_df
