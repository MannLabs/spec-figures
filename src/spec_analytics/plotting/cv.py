"""CV-distribution plots: violin, ECDF, and combined stacked bars.

Extracted from _core.py (REFACTOR_PLAN.md step 5); behaviour unchanged. Heavy plotting deps (matplotlib, seaborn, scipy, sklearn) are imported lazily inside the functions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._style import (PALETTE_SINGLE, PALETTE_CV_CATEGORIES, _hide_top_right_spines, _annotate_stacked_bar, _LEVEL_COLS,
                     _resolve_panel_size)
from ..stats import _compute_cv_table


def plot_cv_vs_abundance(
    df, sample_info, *, level='precursor', group_col='condition2',
    group_order=None, palette=None, n_bins=10, complete_only=True,
    show_iqr=True, cv_threshold=20.0, figsize=None, title=None,
    label_fontsize=10, tick_fontsize=10, legend_fontsize=8, ax=None,
    verbose=True,
):
    """Median CV against abundance, in bins, one line per condition.

    A median CV is one number for a distribution that is strongly
    abundance-dependent: measurement scatter is the quadrature sum of
    ion-statistics noise, which scales as 1/sqrt(intensity), and an
    intensity-independent technical term. This panel separates them — the
    high-abundance end estimates the technical floor alone, and the slope toward
    low abundance is the counting term.

    That makes it the right panel for asking whether one method's precision
    advantage is real across the range or confined to abundant features, which a
    median cannot answer.

    **Complete cases only** by default (`complete_only=True`): entities
    quantified in *every* replicate of their condition. A CV over a subset of
    replicates mixes measurement scatter with detection sporadicity, and telling
    those apart is the whole point here. Turning it off inflates the
    low-abundance end with entities that were simply missed.

    Bins are **per condition** quantiles of that condition's own intensity
    distribution, and each point is drawn at its bin's median intensity. Shared
    absolute bin edges would compare like abundance for like, but absolute
    intensity does not transfer between acquisition methods, so a shared grid
    would silently align different physical amounts. The consequence to keep in
    mind: two curves at the same x are at the same *measured* intensity, which is
    only the same amount of peptide if the methods respond alike.

    :returns: ``(fig, ax, source_df)`` with the plotted medians and quartiles.
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

    rows = []
    for group in groups:
        runs = list(sample_info.loc[sample_info[group_col] == group, 'run'])
        if not runs:
            continue
        wide = (work[work['run'].isin(runs)]
                .pivot_table(index=id_col, columns='run', values=val_col,
                             aggfunc='max')
                .reindex(columns=runs))
        n_obs = wide.notna().sum(axis=1)
        wide = wide[n_obs == len(runs)] if complete_only else wide[n_obs >= 3]
        if wide.empty:
            continue

        cv = (wide.std(axis=1, ddof=1) / wide.mean(axis=1) * 100)
        intensity = np.log10(wide.mean(axis=1))
        frame = pd.DataFrame({'cv': cv, 'log10_intensity': intensity}).dropna()
        # qcut on the condition's own distribution — see the docstring.
        frame['bin'] = pd.qcut(frame['log10_intensity'], n_bins, labels=False,
                              duplicates='drop')
        # Columns deliberately not named 'median'/'q1': on a Series row those
        # shadow DataFrame methods, and `row.median` then returns the method.
        stats = frame.groupby('bin').agg(
            x=('log10_intensity', 'median'), cv_median=('cv', 'median'),
            cv_q1=('cv', lambda s: s.quantile(0.25)),
            cv_q3=('cv', lambda s: s.quantile(0.75)), n_entities=('cv', 'size'))

        ax.plot(stats['x'], stats['cv_median'], marker='o', markersize=3.5,
                lw=1.6, color=palette[group], label=str(group), zorder=3)
        if show_iqr:
            ax.fill_between(stats['x'], stats['cv_q1'], stats['cv_q3'],
                            color=palette[group], alpha=0.15, lw=0, zorder=1)
        rows.extend({'condition': group, 'bin': int(b),
                     'log10_mean_intensity': float(r['x']),
                     'median_cv_pct': float(r['cv_median']),
                     'q1_cv_pct': float(r['cv_q1']),
                     'q3_cv_pct': float(r['cv_q3']),
                     'n': int(r['n_entities'])}
                    for b, r in stats.iterrows())
        if verbose:
            print(f'    {group}: {len(frame):,} complete {level}s, median CV '
                  f'{stats["cv_median"].iloc[-1]:.1f}% in the top bin -> '
                  f'{stats["cv_median"].iloc[0]:.1f}% in the bottom')

    if cv_threshold:
        ax.axhline(cv_threshold, ls=':', color='#d62728', lw=1.2, zorder=0)
    ax.set_xlabel(f'log₁₀ mean {level} intensity', fontsize=label_fontsize)
    ax.set_ylabel(f'{level.capitalize()} CV [%]', fontsize=label_fontsize)
    ax.set_ylim(0, None)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(frameon=False, fontsize=legend_fontsize, loc='upper right')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_cv_violin(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    hue_col=None,
    group_order=None,
    cv_threshold=0.2,
    min_values_for_cv=3,
    figsize=(10, 6),
    palette=None,
    ylim=None,
    as_percent=False,
    title=None,
    y_label=None,
    show_threshold=False,
    show_median_label=True,
    median_label_loc='below',
    median_label_color='black',
    median_label_fontsize=10,
    box_facecolor='#4d4d4d',
    box_edgecolor='black',
    box_width=0.12,
    median_color='white',
    bold_xticklabels=False,
    legend_fontsize=10,
    ax=None,
):
    """
    Violin plot of CV distribution per replicate group.

    Styling follows the house style: per-group fill from `palette` (a list
    cycled over the groups, or a dict keyed by group label), a dark-grey inner
    box with a white median line (no red), and the per-group median CV printed
    in bold below each violin (`median_label_loc='below'`; use `'inline'` for
    the old in-violin label, `'none'` to suppress).

    `as_percent=True` plots CV in percent (0-100) and labels the axis [%].
    `group_order` fixes the x order; `ax` draws onto an existing axis (e.g. for
    stacked PG/peptide panels). `show_threshold` (off by default) adds the red
    dashed CV-threshold guide.

    `hue_col=` splits each x category into side-by-side violins by a second
    sample_info column (e.g. instrument). Empty (group, hue) cells are skipped.

    Returns (fig, ax, cv_stats) where `cv_stats` is a DataFrame with one row
    per group (or per (group, hue) when `hue_col` is set) containing
    n_total, n_CV<10%, n_CV<20%, %_CV<20%, median_CV.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    palette = palette if palette is not None else PALETTE_SINGLE
    palette_is_dict = isinstance(palette, dict)
    cv_table = _compute_cv_table(df, sample_info, level, group_col,
                                 min_values_for_cv, hue_col=hue_col)

    stats_keys = ['group'] + (['hue'] if hue_col else [])
    cv_stats_rows = []
    for keys, sub in cv_table.groupby(stats_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(sub)
        if n == 0:
            continue
        row = {'group': keys[0]}
        if hue_col:
            row['hue'] = keys[1]
        row.update({
            'n_total': n,
            'n_CV<10%': int((sub['cv'] < 0.10).sum()),
            'n_CV<20%': int((sub['cv'] < cv_threshold).sum()),
            '%_CV<20%': float((sub['cv'] < cv_threshold).mean() * 100),
            'median_CV': float(sub['cv'].median()),
            'mean_CV': float(sub['cv'].mean()),
        })
        cv_stats_rows.append(row)
    cv_stats = pd.DataFrame(cv_stats_rows)

    scale = 100.0 if as_percent else 1.0
    cv_table = cv_table.copy()
    cv_table['cv_plot'] = cv_table['cv'] * scale
    if ylim is None:
        ylim = (0, 80) if as_percent else (0, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    if hue_col is None:
        groups = (group_order if group_order is not None
                  else list(cv_table['group'].unique()))
        if palette_is_dict:
            color_map = {g: palette[g] for g in groups}
        else:
            color_map = {g: palette[i % len(palette)]
                         for i, g in enumerate(groups)}
        sns.violinplot(
            data=cv_table, x='group', y='cv_plot', order=groups,
            hue='group', hue_order=groups, palette=color_map,
            ax=ax, inner=None, cut=0, density_norm='width', legend=False,
        )
        sns.boxplot(
            data=cv_table, x='group', y='cv_plot', order=groups, ax=ax,
            width=box_width, showcaps=False,
            boxprops={'facecolor': box_facecolor, 'edgecolor': box_edgecolor,
                      'linewidth': 0.8},
            whiskerprops={'color': box_edgecolor, 'linewidth': 1},
            medianprops={'color': median_color, 'linewidth': 2},
            showfliers=False,
        )
    else:
        # Use only the hues actually present so seaborn doesn't reserve dodge
        # slots for absent (group, hue) combinations.
        hues = list(cv_table['hue'].drop_duplicates())
        groups = (group_order if group_order is not None
                  else list(cv_table['group'].unique()))
        if palette_is_dict:
            color_map = {h: palette[h] for h in hues}
        else:
            color_map = {h: palette[i % len(palette)]
                         for i, h in enumerate(hues)}
        sns.violinplot(
            data=cv_table, x='group', y='cv_plot', order=groups,
            hue='hue', palette=color_map,
            ax=ax, inner=None, cut=0, density_norm='width',
            hue_order=hues, dodge=True,
        )
        # Same inner box as the no-hue branch (dark box, white median) rather
        # than seaborn's quartile lines, so hue-split panels match the house
        # style. Drawn one cell at a time at explicit dodge positions: a dodged
        # seaborn boxplot mis-places boxes when (group, hue) cells are missing,
        # whereas positions computed here simply skip an absent cell.
        # `manage_ticks=False` keeps boxplot from resetting the categorical axis.
        slot = 0.8 / len(hues)          # seaborn's default violin width is 0.8
        for group_index, grp in enumerate(groups):
            for hue_index, hue_value in enumerate(hues):
                cell = cv_table.loc[(cv_table['group'] == grp)
                                    & (cv_table['hue'] == hue_value),
                                    'cv_plot'].dropna()
                if cell.empty:
                    continue
                offset = (hue_index - (len(hues) - 1) / 2) * slot
                ax.boxplot(
                    [cell.to_numpy()], positions=[group_index + offset],
                    widths=box_width * slot / 0.8, patch_artist=True,
                    showcaps=False, showfliers=False, manage_ticks=False,
                    boxprops={'facecolor': box_facecolor,
                              'edgecolor': box_edgecolor, 'linewidth': 0.8},
                    whiskerprops={'color': box_edgecolor, 'linewidth': 1},
                    medianprops={'color': median_color, 'linewidth': 2},
                )

    if show_threshold:
        ax.axhline(cv_threshold * scale, color='red', linestyle='--',
                   linewidth=1.5, label=f'CV = {cv_threshold * 100:.0f}%')

    if show_median_label and hue_col is None and median_label_loc != 'none':
        y_below = ylim[0] - 0.06 * (ylim[1] - ylim[0])
        for i, grp in enumerate(groups):
            row = cv_stats[cv_stats['group'] == grp]
            if row.empty:
                continue
            # The label is always a percentage, whatever the axis units, but it
            # has to be POSITIONED in plot units. Reading `med` for both is the
            # bug this splits apart: with as_percent=False a median CV of 0.066
            # printed as f'{med:.1f}%' came out "0.1%".
            median_cv = float(row['median_CV'].iloc[0])
            med = median_cv * scale
            med_pct = median_cv * 100.0
            if median_label_loc == 'below':
                ax.annotate(
                    f'{med_pct:.1f}%', xy=(i, y_below), ha='center', va='top',
                    fontsize=median_label_fontsize, fontweight='bold',
                    color=median_label_color, annotation_clip=False,
                )
            else:  # 'inline'
                ax.annotate(
                    f'{med_pct:.1f}%', xy=(i + 0.15, med), ha='left', va='center',
                    fontsize=median_label_fontsize, fontweight='bold',
                    color=median_label_color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.8),
                )

    if y_label is None:
        y_label = 'CV [%]' if as_percent else 'Coefficient of Variation'
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylim(ylim)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             fontweight='bold' if bold_xticklabels else 'normal')
    ax.tick_params(labelsize=10)
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight='bold')

    if hue_col is not None:
        ax.legend(fontsize=legend_fontsize, frameon=False,
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    elif show_threshold:
        ax.legend(fontsize=legend_fontsize, loc='upper right', frameon=False)

    sns.despine(ax=ax)
    if created_fig:
        plt.tight_layout()
    return fig, ax, cv_stats


def plot_cv_ecdf(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    group_order=None,
    palette=None,
    cv_threshold=0.2,
    min_values_for_cv=3,
    figsize=(3.5, 3),
    xlim=(0, 60),
    title=None,
    y_label='Cumulative fraction',
    x_label='CV [%]',
    linestyle='-',
    linewidth=1.5,
    annotate_median=True,
    legend_loc='lower right',
    legend_fontsize=8,
    ax=None,
):
    """
    Empirical CDF of per-feature CVs, one step curve per replicate group.

    CVs are plotted in percent. With `annotate_median=True` the per-group
    median CV is appended to each legend entry (e.g. 'C18 (6.2%)'), so the
    median reads directly off the legend; a dotted red guide marks
    `cv_threshold`. House-style tick density: y major 0.1 / minor 0.05; x major
    every 5%.

    `palette` may be a list (cycled over `group_order`) or a dict keyed by
    group label. Pass `ax` to draw onto an existing axis and `linestyle='--'`
    to overlay a second level (e.g. peptide dashed over protein solid).

    Returns (fig, ax, cv_stats) with one row per group
    (group, median_CV, n_total, %_CV<thr).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    palette = palette if palette is not None else PALETTE_SINGLE
    palette_is_dict = isinstance(palette, dict)
    cv_table = _compute_cv_table(df, sample_info, level, group_col,
                                 min_values_for_cv)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    groups = (group_order if group_order is not None
              else list(cv_table['group'].unique()))
    cv_stats_rows = []
    for i, grp in enumerate(groups):
        vals = (cv_table.loc[cv_table['group'] == grp, 'cv']
                .dropna().sort_values().to_numpy() * 100.0)
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        y = np.arange(1, vals.size + 1) / vals.size
        color = palette[grp] if palette_is_dict else palette[i % len(palette)]
        label = f'{grp} ({med:.1f}%)' if annotate_median else str(grp)
        ax.plot(vals, y, drawstyle='steps-post', color=color,
                linewidth=linewidth, linestyle=linestyle, label=label)
        cv_stats_rows.append({
            'group': grp, 'median_CV': med / 100.0, 'n_total': int(vals.size),
            '%_CV<20%': float((vals < cv_threshold * 100).mean() * 100),
        })

    ax.axvline(cv_threshold * 100, color='red', linestyle=':', linewidth=1)
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.tick_params(labelsize=10)
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=legend_fontsize, frameon=False, loc=legend_loc)
    _hide_top_right_spines(ax)
    if created_fig:
        plt.tight_layout()
    return fig, ax, pd.DataFrame(cv_stats_rows)


def plot_cv_stacked_bar_combined(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    hue_col=None,
    min_values_for_cv=3,
    figsize=(7, 6),
    palette=None,
    show_values=True,
    value_format='both',
    bar_width=None,
    bar_gap=0.0,
    title=None,
):
    """
    Stacked bars showing the CV distribution per replicate group at a single
    quantitation level, split into < 10% / 10-20% / >= 20% bins.

    `hue_col=` adds a side-by-side stack per (group, hue). Empty (group, hue)
    cells are skipped silently.

    Returns (fig, ax, cv_counts) where `cv_counts` has columns
    'group', ('hue' if hue_col), 'CV < 10%', 'CV 10-20%', 'CV >= 20%', 'Total'.
    """
    import matplotlib.pyplot as plt

    if level not in _LEVEL_COLS:
        raise ValueError(
            f"level must be one of {sorted(_LEVEL_COLS)}, got {level!r}"
        )
    palette = palette if palette is not None else PALETTE_CV_CATEGORIES
    fig, ax = plt.subplots(figsize=figsize)

    cv_table = _compute_cv_table(df, sample_info, level, group_col,
                                 min_values_for_cv, hue_col=hue_col)
    keys = ['group'] + (['hue'] if hue_col else [])
    cv_count_cols = (['group']
                     + (['hue'] if hue_col else [])
                     + ['CV < 10%', 'CV 10-20%', 'CV >= 20%', 'Total'])
    rows = []
    if not cv_table.empty:
        for partition, sub in cv_table.groupby(keys, sort=False):
            if not isinstance(partition, tuple):
                partition = (partition,)
            cv = sub['cv']
            row = {'group': partition[0]}
            if hue_col:
                row['hue'] = partition[1]
            row.update({
                'CV < 10%': int((cv < 0.10).sum()),
                'CV 10-20%': int(((cv >= 0.10) & (cv < 0.20)).sum()),
                'CV >= 20%': int((cv >= 0.20).sum()),
                'Total': len(cv),
            })
            rows.append(row)
    cv_counts = pd.DataFrame(rows, columns=cv_count_cols)

    categories = ['CV < 10%', 'CV 10-20%', 'CV >= 20%']
    if cv_counts.empty:
        warnings.warn(
            f'plot_cv_stacked_bar_combined: no (group, hue) partition has at '
            f'least min_values_for_cv={min_values_for_cv} replicates — '
            f'rendering an empty plot.'
        )
        groups = list(sample_info[group_col].dropna().unique())
    else:
        groups = list(cv_counts['group'].drop_duplicates())
    x = np.arange(len(groups))

    # Default bar_width depends on layout: 0.6 for single, 0.8 for hue cluster.
    eff_bar_width = bar_width if bar_width is not None else (
        0.8 if hue_col else 0.6
    )

    if hue_col is None:
        bottom = np.zeros(len(groups))
        cv_counts_idx = cv_counts.set_index('group').reindex(groups)
        for cat, color in zip(categories, palette):
            values = cv_counts_idx[cat].astype(float).fillna(0).to_numpy()
            bars = ax.bar(x, values, eff_bar_width, bottom=bottom, label=cat,
                          color=color, edgecolor='black', linewidth=0.5)
            if show_values:
                _annotate_stacked_bar(ax, bars, values, bottom,
                                      cv_counts_idx['Total'].astype(float).fillna(0).to_numpy(),
                                      value_format)
            bottom += values
        for j, total in enumerate(cv_counts_idx['Total'].astype(float).fillna(0).to_numpy()):
            if total > 0:
                ax.annotate(f'n={int(total):,}', xy=(x[j], total),
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=10)
    else:
        hues = list(cv_counts['hue'].drop_duplicates())
        n_hue = len(hues)
        usable = max(eff_bar_width - bar_gap * max(n_hue - 1, 0), 0.0)
        sub_width = usable / max(n_hue, 1)
        offsets = (np.arange(n_hue) - (n_hue - 1) / 2.0) * (sub_width + bar_gap)
        for hi, hue in enumerate(hues):
            cv_h = (cv_counts[cv_counts['hue'] == hue]
                              .set_index('group').reindex(groups))
            bottom = np.zeros(len(groups))
            for ci, (cat, color) in enumerate(zip(categories, palette)):
                values = cv_h[cat].astype(float).fillna(0).to_numpy()
                # Only the first hue gets the colour-category legend entry.
                label = cat if hi == 0 else None
                bars = ax.bar(x + offsets[hi], values, sub_width,
                              bottom=bottom, color=color,
                              edgecolor='black', linewidth=0.5,
                              label=label)
                if show_values:
                    _annotate_stacked_bar(ax, bars, values, bottom,
                                          cv_h['Total'].astype(float).fillna(0).to_numpy(),
                                          value_format)
                bottom += values
            # n= label per hue stack.
            for j, total in enumerate(cv_h['Total'].astype(float).fillna(0).to_numpy()):
                if total > 0:
                    ax.annotate(f'n={int(total):,}',
                                xy=(x[j] + offsets[hi], total),
                                ha='center', va='bottom',
                                fontsize=7, fontweight='bold')
            # Hue label below the x tick (small, in the bar's middle x position).
            for j, grp in enumerate(groups):
                if cv_h['Total'].fillna(0).iloc[j] > 0:
                    ax.text(x[j] + offsets[hi], -0.02, str(hue),
                            ha='center', va='top', rotation=90,
                            fontsize=7, transform=ax.get_xaxis_transform(),
                            clip_on=False)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=10)

    ax.set_title(title if title else f'{level.capitalize()}-level CVs',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=False, fontsize=9, borderaxespad=0)

    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax, cv_counts
