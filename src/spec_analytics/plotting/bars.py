"""Bar / box / scatter count plots: plot_bar, plot_boxplot_with_points, plot_median_scatter, plot_overlapping_bars."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from ._style import (PALETTE, PALETTE_DOUBLE, _hide_top_right_spines, _annotate_bar_values,
                     _replicate_point_size, _area_to_diameter, _resolve_panel_size)
from ..stats import _annotate_significance
BAR_WIDTH_IN = 0.38
BAR_PITCH_IN = 0.50
BAR_GROUP_GAP_IN = 0.28
def set_bar_geometry(bar_in=None, unit_in=None, group_gap_in=None):
    """Set the house bar geometry globally, in inches. Returns the new triple.
    The companion of ``set_default_panel_size`` and ``set_replicate_point_scale``:
    one call changes every bar panel drawn afterwards, so a figure series stays
    uniform without threading the numbers through call sites::
        core.set_bar_geometry(bar_in=0.30, unit_in=0.42)
    Arguments left as None are unchanged. The values are read at call time by
    ``fix_bar_geometry`` and ``plot_grouped_bars``, so this applies to every
    panel drawn after it is called.
    """
    global BAR_WIDTH_IN, BAR_PITCH_IN, BAR_GROUP_GAP_IN
    if bar_in is not None:
        BAR_WIDTH_IN = float(bar_in)
    if unit_in is not None:
        BAR_PITCH_IN = float(unit_in)
    if group_gap_in is not None:
        BAR_GROUP_GAP_IN = float(group_gap_in)
    if BAR_WIDTH_IN > BAR_PITCH_IN:
        warnings.warn(f'bar_in ({BAR_WIDTH_IN}) exceeds unit_in ({BAR_PITCH_IN}); '
                      'adjacent bars will overlap', stacklevel=2)
    return (BAR_WIDTH_IN, BAR_PITCH_IN, BAR_GROUP_GAP_IN)
def fix_bar_geometry(fig, ax, *, bar_in=None, unit_in=None, h_in=None):
    """Rescale the axes so one x unit is `unit_in`, then set bars to `bar_in`.
    Call after ``tight_layout()``. Setting only the bar width would leave the
    *gap* determined by whatever axes width tight_layout happened to produce, so
    panels with different bar counts would still differ visibly in spacing.
    Rescaling the axes first pins the pitch, which pins the gap; the figure width
    moves by the same amount, so the margins tight_layout measured survive.
    ``bbox_inches='tight'`` crops without scaling, so a saved PDF keeps these
    inch values.
    ``h_in`` additionally pins the data-area HEIGHT. Pass it whenever several
    panels of one figure have to share a plot frame: pinning ``figsize`` does not
    achieve that, because a bar panel with rotated x labels and an ECDF panel
    with short ones inherit different data-area heights from the same figure
    height. Left as None the height is untouched, which is the historical
    behaviour.
    ``bar_in`` / ``unit_in`` default to the module globals, read at call time so
    ``set_bar_geometry`` applies.
    """
    from ._style import set_axes_size_inches
    bar_in = BAR_WIDTH_IN if bar_in is None else float(bar_in)
    unit_in = BAR_PITCH_IN if unit_in is None else float(unit_in)
    x0, x1 = ax.get_xlim()
    set_axes_size_inches(fig, ax, w_in=(x1 - x0) * unit_in, h_in=h_in)
    w_data = bar_in / unit_in
    for patch in ax.patches:
        centre = patch.get_x() + patch.get_width() / 2
        patch.set_width(w_data)
        patch.set_x(centre - w_data / 2)
def _legend_above(ax, handles, labels, fontsize, ncol=2):
    """House bar-plot key: one compact horizontal row just above the axes.
    Above rather than inside because a bar chart has no reliably empty corner —
    the tallest bar moves with the data, so a key placed in last month's gap
    lands on this month's bar. Above the axes it never collides, costs no plot
    area, and reads before the bars rather than competing with them.
    Kept tight on purpose: short handles, small pads, no frame. Default legend
    spacing here is about three times as wide and starts to read as a title.
    """
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.0),
              ncol=ncol, frameon=False, fontsize=fontsize, handlelength=0.8,
              handleheight=0.9, handletextpad=0.35, borderpad=0.0,
              columnspacing=0.8)
_TOTAL_LABEL_OFFSET = 0.045
def plot_bar(
    df,
    x_col,
    y_col,
    *,
    hue_col=None,
    figsize=None,
    palette=None,
    y_label=None,
    x_label=None,
    title=None,
    ylim=None,
    bar_width=0.8,
    bar_gap=0.0,
    show_values=True,
    value_format='int',
    value_fontsize=None,
    show_points=False,
    point_size=None,
    point_alpha=0.75,
    point_color='black',
    label_fontsize=11,
    tick_fontsize=10,
    legend_fontsize=10,
    edgecolor='black',
    linewidth=1.0,
):
    """Bar plot with one bar per category (mean of `y_col` per `x_col`).
    Overlaid replicate points (``show_points=True``) default to the same house
    style as ``plot_overlapping_bars``: black with a thin white outline
    (``point_color='black'``), ``point_alpha=0.75``, and sized automatically to
    the number of bars (``point_size=None`` -> ``_replicate_point_size``). Pass
    ``point_color='match'`` to instead fill points in their bar's colour, or an
    explicit ``point_size`` to override.
    Pass `hue_col=` to split each x-category into side-by-side bars by a
    second column (e.g. `x_col='condition2', hue_col='condition1'` for
    instrument-grouped bars). Categories with no matching rows for a given
    hue value are simply skipped.
    `value_fontsize`: font size of the per-bar value labels; defaults to
    `label_fontsize`. Drop it to ~7 (house inline-label size) for dense
    panels where the default overlaps.
    `bar_width`: total cluster width per x category, in axis units (default
    0.8). The remaining 1 - `bar_width` is gap between adjacent x categories.
    `bar_gap`: extra gap between hue bars within a cluster, in axis units
    (default 0.0 — bars touch each other). Increase for cleaner separation,
    e.g. `bar_gap=0.02`.
    """
    import matplotlib.pyplot as plt
    palette = palette if palette is not None else PALETTE
    vfont = value_fontsize if value_fontsize is not None else label_fontsize
    fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    cats = list(pd.unique(df[x_col]))
    xs = np.arange(len(cats))
    if hue_col is None:
        plot_df = df.groupby(x_col, as_index=False)[y_col].mean()
        plot_df = plot_df.set_index(x_col).reindex(cats).reset_index()
        values = plot_df[y_col].values
        bar_colors = [palette[i % len(palette)] for i in range(len(cats))]
        bars = ax.bar(xs, values, width=bar_width, color=bar_colors,
                      edgecolor=edgecolor, linewidth=linewidth)
        if show_points:
            psize = point_size if point_size is not None else _replicate_point_size(len(cats))
            for i, cat in enumerate(cats):
                cat_data = df.loc[df[x_col] == cat, y_col].dropna().values
                if not len(cat_data):
                    continue
                jitter = np.random.uniform(-bar_width * 0.3, bar_width * 0.3, len(cat_data))
                pt_color = bar_colors[i] if point_color == 'match' else point_color
                ax.scatter(xs[i] + jitter, cat_data, s=psize, color=pt_color,
                           alpha=point_alpha, edgecolor='white', linewidth=0.5, zorder=3)
        if show_values:
            _annotate_bar_values(ax, bars, values, value_format, vfont)
        max_val = (df[y_col].max() if show_points else (max(values) if len(values) else 0))
    else:
        hues = list(pd.unique(df[hue_col]))
        n_hue = len(hues)
        usable = max(bar_width - bar_gap * max(n_hue - 1, 0), 0.0)
        sub_width = usable / max(n_hue, 1)
        offsets = (np.arange(n_hue) - (n_hue - 1) / 2.0) * (sub_width + bar_gap)
        hue_colors = [palette[i % len(palette)] for i in range(n_hue)]
        max_val = 0
        for hi, hue in enumerate(hues):
            sub = df[df[hue_col] == hue]
            means = sub.groupby(x_col, as_index=False)[y_col].mean()
            vals = means.set_index(x_col).reindex(cats)[y_col].values
            present_mask = ~np.isnan(vals)
            if not present_mask.any():
                continue
            bars = ax.bar(xs[present_mask] + offsets[hi], vals[present_mask],
                          width=sub_width, color=hue_colors[hi],
                          edgecolor=edgecolor, linewidth=linewidth,
                          label=str(hue))
            if show_values:
                _annotate_bar_values(ax, bars, vals[present_mask], value_format,
                                     vfont)
            if show_points:
                psize = (point_size if point_size is not None
                         else _replicate_point_size(len(cats) * n_hue))
                for i, cat in enumerate(cats):
                    if not present_mask[i]:
                        continue
                    cat_data = sub.loc[sub[x_col] == cat, y_col].dropna().values
                    if not len(cat_data):
                        continue
                    jitter = np.random.uniform(-sub_width * 0.25, sub_width * 0.25,
                                               len(cat_data))
                    pt_color = hue_colors[hi] if point_color == 'match' else point_color
                    ax.scatter(xs[i] + offsets[hi] + jitter, cat_data,
                               s=psize, color=pt_color, alpha=point_alpha,
                               edgecolor='white', linewidth=0.5, zorder=3)
            max_val = max(max_val, np.nanmax(vals[present_mask]))
        ax.legend(fontsize=legend_fontsize, frameon=False,
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_ylabel(y_label if y_label else y_col, fontsize=label_fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold')
    ax.set_ylim(bottom=0)
    if ylim is not None:
        if isinstance(ylim, (int, float)):
            ax.set_ylim(top=ylim)
        else:
            ax.set_ylim(ylim)
    else:
        ax.set_ylim(top=max_val * 1.15 if max_val else 1)
    if df[y_col].max() > 1000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax
def plot_boxplot_with_points(
    df,
    x_col,
    y_col,
    *,
    hue_col=None,
    figsize=None,
    palette=None,
    y_label=None,
    x_label=None,
    title=None,
    ylim=None,
    ymin=None,
    box_width=0.6,
    show_points=True,
    point_size=None,
    point_alpha=1.0,
    point_color='match',
    point_edgecolor='black',
    point_edge_linewidth=0.4,
    xtick_rotation=45,
    showfliers=False,
    show_median_labels=False,
    median_label_decimals=1,
    median_label_fontsize=8,
    label_fontsize=10,
    tick_fontsize=10,
    legend_fontsize=8,
    significance_test=None,
    significance_correction=None,
    significance_show_ns=False,
    significance_pairs=None,
    significance_fontsize=12,
):
    """Boxplot with optional jittered-point overlay.
    `show_points=True` (default) overlays jittered raw points; set False for
    a plain boxplot. `showfliers=True` enables matplotlib's default outlier
    points on the whiskers (off by default — useful when `show_points=False`
    so the distribution tails remain visible).
    `show_median_labels=True` annotates each box's median above it in the
    box's colour. `median_label_decimals` controls precision (default 1).
    Y-axis limits:
      ylim: scalar -> y-axis top; (lo, hi) tuple -> full range.
      ymin: scalar -> y-axis bottom only (applied after `ylim`, so it can
            override the lower edge regardless of how `ylim` was specified).
    Pairwise significance testing (only when `hue_col` is None or matches
    `x_col`):
      significance_test:
        None        -> no test (default).
        'welch'     -> Welch's t-test (unequal variance) — recommended for
                       proteomics QC summaries (counts, log-intensities)
                       where the data is approximately normal but variances
                       may differ between conditions. Standard at small n.
        'ttest'     -> Student's t-test (equal variance assumed).
        'mwu'       -> Mann-Whitney U (non-parametric; use when normality is
                       in doubt — e.g. heavily skewed distributions).
      significance_correction: None | 'bonferroni' | 'holm' | 'fdr_bh' (etc).
        Multiple-comparisons correction applied across the family of pairwise
        p-values. None by default; recommended 'holm' when comparing >2
        groups.
      significance_show_ns: include 'ns' brackets for non-significant pairs
        (default False — only stars are drawn).
      significance_pairs: optional list of (i, j) integer-index tuples to
        restrict which pairs are tested (default = all pairwise).
      Stars use the conventional thresholds:
        p < 0.05  -> *
        p < 0.01  -> **
        p < 0.001 -> ***
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = palette if palette is not None else PALETTE
    palette_is_dict = isinstance(palette, dict)
    fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    same_col = hue_col is not None and x_col == hue_col
    n_x = df[x_col].nunique()
    box_kwargs = dict(
        x=x_col, y=y_col, data=df, ax=ax, width=box_width, showfliers=showfliers,
        saturation=1.0,
        boxprops={'edgecolor': 'black', 'linewidth': 1},
        whiskerprops={'color': 'black', 'linewidth': 1},
        capprops={'color': 'black', 'linewidth': 1},
        medianprops={'color': 'black', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markersize': 3,
                    'markerfacecolor': 'black', 'markeredgecolor': 'none',
                    'alpha': 0.5},
    )
    if hue_col and not same_col:
        n_hue = df[hue_col].nunique()
        box_kwargs['hue'] = hue_col
        box_kwargs['palette'] = palette if palette_is_dict else palette[:n_hue]
    else:
        box_kwargs['hue'] = x_col
        box_kwargs['palette'] = palette if palette_is_dict else palette[:n_x]
        box_kwargs['legend'] = False
    sns.boxplot(**box_kwargs)
    if show_points:
        if point_size is not None:
            psize = point_size
        else:
            n_bars = n_x * (df[hue_col].nunique() if (hue_col and not same_col) else 1)
            psize = _area_to_diameter(_replicate_point_size(n_bars))
        strip = dict(x=x_col, y=y_col, data=df, ax=ax, size=psize,
                     alpha=point_alpha, jitter=True,
                     edgecolor=point_edgecolor, linewidth=point_edge_linewidth)
        if point_color == 'match':
            if hue_col and not same_col:
                strip.update(hue=hue_col, dodge=True,
                             palette=palette if palette_is_dict else palette,
                             legend=False)
            else:
                strip.update(hue=x_col,
                             palette=palette if palette_is_dict else palette[:n_x],
                             legend=False)
        else:
            strip['color'] = point_color
            if hue_col and not same_col:
                strip.update(hue=hue_col, dodge=True,
                             palette=[point_color] * df[hue_col].nunique(), legend=False)
        sns.stripplot(**strip)
    if show_median_labels and not (hue_col and not same_col):
        x_levels = list(pd.unique(df[x_col]))
        tick_locs = ax.get_xticks()
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        if not tick_labels or any(not lab for lab in tick_labels):
            tick_labels = [str(lv) for lv in x_levels]
        for k, lab in enumerate(tick_labels):
            sub = df.loc[df[x_col].astype(str) == str(lab), y_col].dropna()
            if sub.empty:
                continue
            med = float(np.median(sub))
            if palette_is_dict:
                color = palette.get(lab, palette.get(str(lab), '#444444'))
            else:
                color = palette[k % len(palette)]
            ax.annotate(f'{med:.{median_label_decimals}f}',
                        xy=(tick_locs[k], med),
                        xytext=(0, 8),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        color=color, fontsize=median_label_fontsize,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.8))
    ax.set_ylabel(y_label if y_label else y_col, fontsize=label_fontsize)
    ax.set_xlabel(x_label if x_label else '', fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold')
    if ylim is not None:
        if isinstance(ylim, (int, float)):
            ax.set_ylim(top=ylim)
        else:
            ax.set_ylim(ylim)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if xtick_rotation != 0:
        plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha='right')
    if df[y_col].max() > 1000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    if hue_col and not same_col:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=legend_fontsize,
                  frameon=False, borderaxespad=0)
    if significance_test is not None:
        if hue_col and not same_col:
            warnings.warn('significance_test ignored when hue_col is set '
                          '(only supported on the simple x-only layout)')
        else:
            _annotate_significance(
                ax, df, x_col, y_col,
                test=significance_test,
                correction=significance_correction,
                show_ns=significance_show_ns,
                pairs=significance_pairs,
                fontsize=significance_fontsize,
            )
    _hide_top_right_spines(ax)
    plt.tight_layout()
    return fig, ax
def plot_median_scatter(
    df,
    x_col,
    y_col,
    *,
    hue_col=None,
    figsize=(5, 5),
    palette=None,
    error='iqr',
    point_size=40,
    point_alpha=0.7,
    jitter_width=0.18,
    median_marker='D',
    median_marker_size=130,
    median_marker_color='match',
    median_marker_edgecolor='black',
    error_color='black',
    error_linewidth=1.5,
    error_capsize=5,
    y_label=None,
    x_label=None,
    title=None,
    ylim=None,
    label_fontsize=11,
    tick_fontsize=10,
    legend_fontsize=10,
    rng_seed=0,
):
    """Per-category scatter of every individual point, with a distinct
    summary marker and a vertical error bar for the spread.
    All replicate values are shown jittered around their x category. The
    summary marker (diamond by default, configurable via `median_marker=`)
    sits at the central-tendency estimate, with a capless vertical line
    spanning the spread interval — deliberately *not* drawn like a boxplot.
    Parameters:
      error: choice of central-tendency + spread.
        'iqr'  -> median, error bar spans Q1..Q3 (default; robust).
        'mad'  -> median, error bar is +/- median absolute deviation.
        'std'  -> mean, error bar is +/- one sample standard deviation
                  (the parametric "mean +/- SD" — preferred for proteomics
                  replicate QC where log2 intensities are roughly normal).
        'sem'  -> mean, error bar is +/- standard error of the mean
                  (= SD / sqrt(n); under-represents uncertainty at small n).
      median_marker: any matplotlib marker code ('D' diamond, 'o' circle,
        's' square, '_' horizontal bar, etc.).
      median_marker_color: 'match' (default) takes the cluster colour so the
        diamond is visually tied to its hue group; pass any matplotlib colour
        ('black', '#444', etc.) to force a fixed colour instead.
      median_marker_edgecolor: edge colour around the diamond (default
        'black' — contrasts with both pale and saturated palette fills).
      error_color: colour of the error-bar line (default 'black').
      error_capsize: width of the horizontal end-caps on the whiskers in
        points (default 5; pass 0 to disable).
    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    palette = palette if palette is not None else PALETTE
    fig, ax = plt.subplots(figsize=figsize)
    categories = list(pd.unique(df[x_col]))
    rng = np.random.default_rng(rng_seed)
    if hue_col is None:
        groups = [(cat, None, palette[i % len(palette)], i, 0.0)
                  for i, cat in enumerate(categories)]
        sub_jitter_width = jitter_width
    else:
        hues = list(pd.unique(df[hue_col]))
        n_hue = len(hues)
        cluster_width = 0.7
        sub_step = cluster_width / max(n_hue, 1)
        offsets = (np.arange(n_hue) - (n_hue - 1) / 2.0) * sub_step
        sub_jitter_width = min(jitter_width, sub_step * 0.4)
        hue_colors = [palette[i % len(palette)] for i in range(n_hue)]
        groups = [
            (cat, hue, hue_colors[hi], i, offsets[hi])
            for i, cat in enumerate(categories)
            for hi, hue in enumerate(hues)
        ]
    seen_hues = set()
    for cat, hue, color, i, dx in groups:
        if hue_col is None:
            vals = df.loc[df[x_col].astype(str) == str(cat), y_col].dropna().to_numpy()
        else:
            mask = ((df[x_col].astype(str) == str(cat))
                    & (df[hue_col].astype(str) == str(hue)))
            vals = df.loc[mask, y_col].dropna().to_numpy()
        if vals.size == 0:
            continue
        x_jitter = rng.uniform(-sub_jitter_width, sub_jitter_width, size=vals.size) + i + dx
        scatter_label = None
        if hue_col is not None and hue not in seen_hues:
            scatter_label = str(hue)
            seen_hues.add(hue)
        ax.scatter(x_jitter, vals, s=point_size, alpha=point_alpha,
                   color=color, edgecolor='white', linewidth=0.5, zorder=2,
                   label=scatter_label)
        if error == 'iqr':
            center = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            err_low, err_high = center - q1, q3 - center
        elif error == 'mad':
            center = float(np.median(vals))
            mad = float(np.median(np.abs(vals - center)))
            err_low = err_high = mad
        elif error == 'std':
            center = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            err_low = err_high = s
        elif error == 'sem':
            center = float(np.mean(vals))
            s = (float(np.std(vals, ddof=1)) / np.sqrt(vals.size)
                 if vals.size > 1 else 0.0)
            err_low = err_high = s
        else:
            raise ValueError(
                f"error must be 'iqr', 'std', 'sem', or 'mad', got {error!r}"
            )
        marker_face = color if median_marker_color == 'match' else median_marker_color
        ax.errorbar([i + dx], [center], yerr=[[err_low], [err_high]],
                    fmt='none', ecolor=error_color,
                    elinewidth=error_linewidth, capsize=error_capsize,
                    capthick=error_linewidth, zorder=3)
        ax.scatter([i + dx], [center], marker=median_marker, s=median_marker_size,
                   color=marker_face,
                   edgecolor=median_marker_edgecolor,
                   linewidth=1.2, zorder=4)
    if hue_col is not None:
        ax.legend(fontsize=legend_fontsize, frameon=False,
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([str(c) for c in categories], rotation=45, ha='right',
                       fontsize=tick_fontsize)
    ax.set_ylabel(y_label if y_label else y_col, fontsize=label_fontsize,
                  fontweight='bold')
    ax.set_xlabel(x_label if x_label else '', fontsize=label_fontsize,
                  fontweight='bold')
    if title:
        ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold')
    if ylim is not None:
        if isinstance(ylim, (int, float)):
            ax.set_ylim(top=ylim)
        else:
            ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if df[y_col].max() > 1000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    sns.despine()
    plt.tight_layout()
    return fig, ax
def plot_overlapping_bars(
    df,
    x_col,
    higher_y_col,
    lower_y_col,
    *,
    hue_col=None,
    higher_label='IDs',
    lower_label='< 20% CV',
    figsize=(8, 6),
    palette=None,
    y_label='Count',
    x_label=None,
    title=None,
    bar_width=0.8,
    bar_gap=0.02,
    show_points=True,
    point_color='black',
    point_size=None,
    point_alpha=0.75,
    label_fontsize=11,
    value_fontsize=None,
    tick_fontsize=10,
    legend_fontsize=7,
    legend_loc='top',
    lighten_factor=0.45,
    constant_bar_width=True,
    ax=None,
):
    """
    Two overlapping bars per (x, hue): a light "total" and a darker subset.
    Typical use: x=condition, higher_y_col='total_protein_groups',
    lower_y_col='PG20', hue='engine'.
    Defaults reproduce the house "total vs CV<20%" look:
    * **Replicate points are always drawn** — black, thin white outline,
      jittered on the light bar. A bar is a mean, and a mean without its points
      hides how many replicates it rests on and whether they agree; there is no
      version of this panel where they should be off. ``show_points=False``
      exists for the rare case where the points genuinely are not per-replicate.
    * **The key sits in one compact row just above the axes**
      (``legend_loc='top'``, the default) — light = ``higher_label`` ("IDs"),
      dark = ``lower_label`` ("< 20% CV"). A bar chart has no reliably empty
      corner, so an in-axes legend collides as soon as the data changes.
      ``'outside'`` parks it to the right, ``'none'`` suppresses it on all but
      one panel of a shared row, and any matplotlib ``loc`` still places it
      inside.
    * **Headroom above the tallest bar** for the value labels, so they are never
      clipped and the legend never overlaps a bar.
    * **Constant bar width in inches** across panels
      (``constant_bar_width=True``, the default): the figure width is set from
      the bar count rather than the bars being stretched to fill a fixed width.
      A three-bar panel is therefore narrower than a five-bar one, but its bars
      are the same size — which is what makes a row of bar panels read as one
      figure. See :func:`fix_bar_geometry`; it is skipped when you pass ``ax``,
      since resizing the figure is not appropriate inside someone else's grid.
    Pass ``point_color=
    'match'`` to instead fill points in their bar's colour, or an explicit
    ``point_size`` to override the auto size.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    palette = palette if palette is not None else PALETTE_DOUBLE
    vfont = value_fontsize if value_fontsize is not None else label_fontsize
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    y_span = float(df[higher_y_col].max() or 0)
    def lighten(color, factor=0.4):
        c = mcolors.to_rgb(color)
        return tuple(1 - (1 - ch) * (1 - factor) for ch in c)
    if legend_loc == 'upper left':
        legend_loc = 'top'
    x_categories = list(df[x_col].unique())
    same_col = hue_col is not None and x_col == hue_col
    if hue_col and not same_col:
        hue_groups = list(df[hue_col].unique())
    else:
        hue_groups = [None]
    n_groups = len(hue_groups)
    group_width = bar_width / n_groups
    x_positions = np.arange(len(x_categories))
    psize = (point_size if point_size is not None
             else _replicate_point_size(len(x_categories) * n_groups))
    summary, legend_handles, legend_labels = [], [], []
    for i, hue in enumerate(hue_groups):
        hue_df = df[df[hue_col] == hue] if (hue_col and not same_col) else df
        offset = (i - (n_groups - 1) / 2) * (group_width + bar_gap)
        positions = x_positions + offset
        for j, x_cat in enumerate(x_categories):
            cat_df = hue_df[hue_df[x_col] == x_cat]
            if cat_df.empty:
                continue
            color = palette[i % len(palette)] if (hue_col and not same_col) else palette[j % len(palette)]
            light = lighten(color, lighten_factor)
            higher = cat_df[higher_y_col].mean()
            lower = cat_df[lower_y_col].mean()
            pos = positions[j]
            ax.bar(pos, higher, width=group_width, color=light, edgecolor='darkgray', linewidth=1.0)
            ax.bar(pos, lower, width=group_width, color=color, edgecolor='black', linewidth=1.0)
            if show_points:
                jitter = np.random.uniform(-group_width * 0.25, group_width * 0.25, len(cat_df))
                pt = color if point_color == 'match' else point_color
                ax.scatter(pos + jitter, cat_df[higher_y_col].values, c=pt,
                           s=psize, alpha=point_alpha, zorder=5,
                           edgecolors='white', linewidths=0.5)
            ax.text(pos, higher + y_span * _TOTAL_LABEL_OFFSET,
                    f'{int(higher):,}', ha='center', va='bottom',
                    fontsize=vfont)
            ax.text(pos, lower, f'{int(lower):,}', ha='center', va='top',
                    fontsize=vfont,
                    color='white' if lower > higher * 0.1 else 'black')
            summary.append({
                'x_category': x_cat,
                'hue_group': hue if (hue_col and not same_col) else 'all',
                'higher_mean': higher, 'lower_mean': lower,
                'higher_std': cat_df[higher_y_col].std(),
                'lower_std': cat_df[lower_y_col].std(),
                'n_samples': len(cat_df),
                'percent_lower': lower / higher * 100 if higher > 0 else 0,
            })
        if hue_col and not same_col:
            color = palette[i % len(palette)]
            light = lighten(color, lighten_factor)
            legend_handles.append(Patch(facecolor=light, edgecolor='darkgray', linewidth=1.0))
            legend_labels.append(str(hue))
            legend_handles.append(Patch(facecolor=color, edgecolor='black', linewidth=1.0))
            legend_labels.append(f'{hue} ({lower_label})')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_categories, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 2, fontweight='bold',
                     pad=legend_fontsize + 12 if legend_loc == 'top' else 6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylim(bottom=0, top=df[higher_y_col].max() * 1.15)
    if not (hue_col and not same_col):
        grey_dark = '#595959'
        grey_light = lighten(grey_dark, 0.55)
        legend_handles = [
            Patch(facecolor=grey_light, edgecolor='darkgray', linewidth=0.8),
            Patch(facecolor=grey_dark, edgecolor='black', linewidth=0.8),
        ]
        legend_labels = [higher_label, lower_label]
        if legend_loc == 'none':
            pass
        elif legend_loc == 'top':
            _legend_above(ax, legend_handles, legend_labels, legend_fontsize)
        elif legend_loc == 'outside':
            ax.legend(legend_handles, legend_labels, loc='upper left',
                      bbox_to_anchor=(1.02, 1), fontsize=legend_fontsize,
                      frameon=False, borderaxespad=0)
        else:
            ax.legend(legend_handles, legend_labels, loc=legend_loc,
                      fontsize=legend_fontsize, frameon=False)
    elif legend_handles:
        if legend_loc == 'none':
            pass
        elif legend_loc == 'top':
            _legend_above(ax, legend_handles, legend_labels, legend_fontsize,
                          ncol=min(len(legend_labels), 4))
        else:
            ax.legend(legend_handles, legend_labels, loc='upper left',
                      bbox_to_anchor=(1.02, 1), fontsize=legend_fontsize,
                      frameon=False, borderaxespad=0)
    _hide_top_right_spines(ax)
    if created_fig:
        plt.tight_layout()
        if constant_bar_width:
            fix_bar_geometry(fig, ax)
    return fig, ax, pd.DataFrame(summary)
def plot_grouped_bars(
    groups,
    heights,
    *,
    colors=None,
    points=None,
    ax=None,
    figsize=None,
    y_label=None,
    ylim=None,
    bar_in=None,
    unit_in=None,
    group_gap_in=None,
    h_in=None,
    constant_bar_width=True,
    legend=True,
    legend_ncol=None,
    legend_loc='best',
    legend_fontsize=None,
    point_size=None,
    point_alpha=0.8,
    point_color='black',
    point_jitter=0.10,
    edgecolor='black',
    linewidth=0.7,
    tick_fontsize=None,
    label_fontsize=None,
):
    """Bars in labelled groups: one x tick per GROUP, colour identifies the bar.
    The construction to reach for whenever the categories on the x axis are of
    two kinds — formats against phases, instruments against methods — where a
    flat row of bars reads as one homogeneous series and silently invites the
    reader to compare across the divide. Grouping puts the distinction in the
    layout, so the panel needs neither a bracket drawn inside the axes nor a
    second label row underneath, both of which cost plot area and get deleted at
    layout time anyway::
        core.plot_grouped_bars(
            [('in solution', ['ISD+', 'ISD']), ('SPEC', ['C18', 'SCX', 'SAX'])],
            heights={'ISD+': 7841, 'ISD': 8579, 'C18': 6154, ...},
            colors={...}, points={'ISD+': [7830, 7855, ...], ...},
            y_label='Protein groups')
    Parameters
    ----------
    groups : list of (group_label, [item, ...])
        Bars are drawn in this order; ``group_label`` becomes the x tick label.
    heights, colors, points : dict keyed by item
        ``heights`` is the bar height. ``colors`` defaults to ``PALETTE`` cycled
        over the items in order. ``points`` is optional per-item replicate
        values, overlaid in the house style (black, thin white edge, jittered).
    bar_in, unit_in, group_gap_in, h_in : float, optional
        Geometry in inches; default to the module globals, so
        ``core.set_bar_geometry(...)`` moves every panel at once. ``group_gap_in``
        is the EXTRA space between groups, on top of one pitch. ``h_in`` pins the
        data-area height so this panel matches its neighbours.
    Returns
    -------
    (fig, ax, positions) where ``positions`` maps each item to its x centre, so a
    caller can annotate a specific bar without recomputing the layout.
    Notes
    -----
    Replicate points are sized from the DRAWN bar width in a second pass, after
    the geometry is fixed — the bar width is not known until then, and a
    hand-set ``s`` is what produces dots that look right in one panel and like
    grit in the next. Pass ``point_size`` to override.
    """
    import matplotlib.pyplot as plt
    from ._style import replicate_point_size, scale_replicate_points
    bar_in = BAR_WIDTH_IN if bar_in is None else float(bar_in)
    unit_in = BAR_PITCH_IN if unit_in is None else float(unit_in)
    group_gap_in = BAR_GROUP_GAP_IN if group_gap_in is None else float(group_gap_in)
    items = [it for _, members in groups for it in members]
    if not items:
        raise ValueError('groups contains no items')
    missing = [it for it in items if it not in heights]
    if missing:
        raise KeyError(f'no height for {missing}')
    if colors is None:
        colors = {it: PALETTE[i % len(PALETTE)] for i, it in enumerate(items)}
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure
    gap_units = group_gap_in / unit_in
    bar_frac = bar_in / unit_in
    x, centres, labels, positions = 0.0, [], [], {}
    dots = []
    for gi, (glabel, members) in enumerate(groups):
        if gi:
            x += gap_units
        first = x
        for item in members:
            ax.bar(x, heights[item], width=bar_frac, color=colors[item],
                   edgecolor=edgecolor, lw=linewidth, zorder=2)
            if points is not None and points.get(item) is not None:
                vals = np.asarray(points[item], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    xs = (x + np.linspace(-point_jitter, point_jitter, vals.size)
                          if vals.size > 1 else np.array([x]))
                    dots.append(ax.scatter(
                        xs, vals, s=point_size or 14,
                        color=(colors[item] if point_color == 'match' else point_color),
                        alpha=point_alpha, linewidth=0.3, edgecolor='white',
                        zorder=5))
            positions[item] = x
            x += 1.0
        centres.append((first + x - 1.0) / 2)
        labels.append(glabel)
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.set_xlim(-0.5 - bar_frac / 2, x - 1.0 + 0.5 + bar_frac / 2)
    if y_label:
        ax.set_ylabel(y_label, fontsize=label_fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(False)
    _hide_top_right_spines(ax)
    if legend:
        from matplotlib.patches import Rectangle
        handles = [Rectangle((0, 0), 1, 1, fc=colors[it], ec=edgecolor,
                             lw=linewidth) for it in items]
        if legend_loc == 'top':
            _legend_above(ax, handles, items, legend_fontsize,
                          ncol=legend_ncol or min(len(items), 4))
        elif legend_loc != 'none':
            ax.legend(handles, items, loc=legend_loc, frameon=False,
                      ncol=legend_ncol or 1, fontsize=legend_fontsize,
                      handlelength=0.9, handletextpad=0.4, columnspacing=0.9,
                      borderpad=0.2)
    if created_fig:
        plt.tight_layout()
        if constant_bar_width:
            fix_bar_geometry(fig, ax, bar_in=bar_in, unit_in=unit_in, h_in=h_in)
    if dots:
        if point_size is None:
            scale_replicate_points(ax, dots)
        else:
            for d in dots:
                d.set_sizes([point_size])
    return fig, ax, positions
