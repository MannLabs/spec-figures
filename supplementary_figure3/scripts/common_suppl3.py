"""Shared machinery for supplementary figure 3 — peptide counts and their CVs."""

import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import spec_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
# Geometry constants live with figure 2 and are imported, never duplicated.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'figure2',
                                                'scripts')))
import common_figure2 as cf                                    # noqa: E402

FIG2_INPUT = _cfg.cross_input('figure2')
OUTDIR = _cfg.output_dir(__file__)
CACHE_DIR = os.path.abspath(_cfg.data_dir(__file__))

QVALUE = cf.QVALUE
MIN_VALUES_FOR_CV = cf.MIN_VALUES_FOR_CV
CV_THRESHOLD = cf.CV_THRESHOLD
FONTSIZE = cf.FONTSIZE
LEGEND_FONTSIZE = cf.LEGEND_FONTSIZE
YLABEL = 'Peptides'
SHARED_YMAX = 160_000
YTICKS = list(range(0, 160_001, 20_000))
# Right axis: summed raw precursor intensity, in units of 1e12.
INTENSITY_SCALE = 1e12
RIGHT_LABEL = 'Summed precursor intensity [$10^{12}$]'
RIGHT_HEADROOM = 1.14
# Value labels off, as in figures 1 and 2. Panel a carries 16 bars x 2 series and
# the numbers collide; they are all in the source data. Set True for a quick look.
SHOW_VALUE_LABELS = False


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
read_peptides = cf.read_peptides
counts_and_cv = cf.counts_and_cv
summarise = cf.summarise


def read_precursor_sums(path, *, runs=None):
    """Summed raw `Precursor.Quantity` per run — the figure's intensity readout.

    **Precursor level, everywhere.** Summing the peptide- or protein-rolled-up
    quantity would double nothing but would make the number depend on the roll-up
    key, so it would move when the peptide definition moves; the precursor sum is
    the one intensity that is a property of the run alone. Filter is
    `read_peptides`', so the bars and the line describe the same set of rows.

    Raw, never normalised: the whole point of the panel is that the conditions
    differ in how much signal survives, and normalising would remove exactly that.
    """
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [c for c in ('Run', 'Precursor.Quantity', 'Q.Value', 'PG.Q.Value',
                           'Decoy') if c in available]
    filters = [('Q.Value', '<=', QVALUE), ('PG.Q.Value', '<=', QVALUE),
               ('Precursor.Quantity', '>', 0)]
    if 'Decoy' in available:
        filters.append(('Decoy', '==', 0))
    d = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    if runs is not None:
        d = d[d['Run'].isin(runs)]
    return (d.groupby('Run', sort=True)['Precursor.Quantity'].sum()
            / INTENSITY_SCALE)

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_grouped_overlapping(ax, categories, series, *,
                             point_size=cf.POINT_SIZE_PLACEHOLDER, seed=0):
    """Grouped overlapping bars: light total, dark CV < 20 % subset.

    `series` is a list of dicts:
        label   legend / source-data text
        method  method identity
        color   method hue
        totals  {category: sequence of per-replicate peptide counts}
        cv20    {category: int}

    Bar slots, widths, group gaps and edge padding all come from
    `common_figure2`, so a supplement panel and its main panel have bars of the
    same width at the same panel width.

    **Only the light bar carries replicate points.** The dark bar is a
    condition-level count computed across the replicates and has no
    per-replicate decomposition — the one bar in the house style that is
    legitimately without points.
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    step = n + cf.GROUP_GAP
    jitter = 0.22 * cf.BAR_FRAC

    heights, bar_points = [], []
    for k, s in enumerate(series):
        for i, cat in enumerate(categories):
            vals = np.asarray(s['totals'][cat], dtype=float)
            centre = i * step + k + 0.5
            mean = float(np.mean(vals))
            ax.bar(centre, mean, cf.BAR_FRAC, color=cf.lighten(s['color']),
                   edgecolor='darkgray', linewidth=0.7, zorder=2)
            ax.bar(centre, s['cv20'][cat], cf.BAR_FRAC, color=s['color'],
                   edgecolor='black', linewidth=0.7, zorder=3)
            bar_points.append(ax.scatter(
                centre + rng.uniform(-jitter, jitter, size=vals.size), vals,
                s=point_size, color='black', alpha=0.85, linewidth=0.3,
                edgecolor='white', zorder=5))
            heights.append((s['label'], cat, mean, s['cv20'][cat]))
            if SHOW_VALUE_LABELS:
                ax.text(centre, mean * 1.02, f'{mean:,.0f}', ha='center',
                        va='bottom', fontsize=6.5, rotation=90)

    centres = [i * step + n / 2 for i in range(len(categories))]
    first_edge = 0.5 - cf.BAR_FRAC / 2
    last_edge = (len(categories) - 1) * step + n - 0.5 + cf.BAR_FRAC / 2
    ax.set_xlim(first_edge - cf.EDGE_PAD, last_edge + cf.EDGE_PAD)
    ax.set_xticks(centres)
    ax.set_xticklabels([str(c) for c in categories], fontsize=FONTSIZE)
    return heights, bar_points


def draw_right_lines(ax_right, categories, series, *,
                     point_size=cf.POINT_SIZE_PLACEHOLDER):
    """Summed precursor intensity as one line per method on the twin axis.

    Same construction as figure 2's right axis: each line runs through the x
    centres of its own method's bars rather than through the group centre, so it
    sits above the bars it belongs to. Drawn as a line, not a third bar, because
    the quantity spans up to 10x within a panel and a bar at 1 % of the axis is
    invisible where a marker is still a marker.

    `series` must be the same list, in the same order, that was passed to
    `draw_grouped_overlapping` — the bar slot is derived from the series index —
    and each entry needs an `intensity` mapping of category to per-replicate
    values. Returns the (label, category, mean) triples for the source data.
    """
    n = len(series)
    step = n + cf.GROUP_GAP
    heights = []
    for k, s in enumerate(series):
        xs = [i * step + k + 0.5 for i in range(len(categories))]
        means = []
        for x, cat in zip(xs, categories):
            vals = np.asarray(s['intensity'][cat], dtype=float)
            means.append(float(np.mean(vals)))
            if vals.size > 1:
                ax_right.scatter(np.full(vals.size, x), vals, s=0.5 * point_size,
                                 color=s['color'], alpha=0.6, linewidth=0.0,
                                 zorder=6)
        ax_right.plot(xs, means, color=s['color'], linewidth=1.6, marker='o',
                      markersize=5.0, markerfacecolor=s['color'],
                      markeredgecolor='white', markeredgewidth=0.7, zorder=7)
        heights.extend((s['label'], cat, m) for cat, m in zip(categories, means))
    return heights


def style_right_axis(ax_left, ax_right, right_max):
    """Zero-based twin axis carrying the intensity line."""
    ax_right.set_ylim(0, right_max * RIGHT_HEADROOM)
    ax_right.set_ylabel(RIGHT_LABEL, fontsize=FONTSIZE)
    ax_right.tick_params(labelsize=FONTSIZE)
    ax_right.tick_params(axis='x', length=0)
    ax_right.spines['top'].set_visible(False)
    # `init_plotting` disables the right spine globally, which on a twinned pair
    # leaves ticks and labels with no axis line to sit on.
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_visible(True)
    ax_right.spines['right'].set_linewidth(
        ax_left.spines['left'].get_linewidth())


def style_axes(ax, *, xlabel, ymax=None, headroom=1.08):
    """House axis styling. `ymax=None` uses the shared limit — see SHARED_YMAX.

    Headroom is 1.08 rather than figure 2's 1.14: the value labels that headroom
    was reserved for are off here, and both keys sit outside the axes.
    """
    ax.set_ylim(0, SHARED_YMAX if ymax is None else ymax * headroom)
    ax.set_yticks(YTICKS if ymax is None else ax.get_yticks())
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(YLABEL, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    # Thousands separators, as on figure 2's protein-group axis. These counts run
    # to six digits, where a bare `160000` is genuinely hard to read at a glance.
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _pos: f'{v:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def metric_key(*, with_intensity=True):
    """Grey key naming the three marks; hue stays reserved for method identity.

    Two bars and a line, matching the panel: light bar = all peptides, dark bar =
    the CV < 20 % subset, line = summed precursor intensity on the right axis.
    """
    key = [Patch(facecolor=cf.lighten(cf.LEGEND_DARK), edgecolor='darkgray',
                 linewidth=0.7, label='Peptides'),
           Patch(facecolor=cf.LEGEND_DARK, edgecolor='black', linewidth=0.7,
                 label='CV < 20 %')]
    if with_intensity:
        key.append(Line2D([0], [0], color=cf.LEGEND_DARK, linewidth=1.6,
                          marker='o', markersize=5.0,
                          markerfacecolor=cf.LEGEND_DARK,
                          markeredgecolor='white', markeredgewidth=0.7,
                          label='Precursor intensity'))
    return key


def finish(fig, ax, bar_points, stem, *, w_in=None):
    """Pin the data area, resize the replicate dots, save PDF + PNG.

    `ax` may be a single axes or the twinned pair; a twinned pair has to be
    repositioned together or the two stop sharing a frame. Replicate dots are
    resized against the first axes, which is the one carrying the bars.
    """
    axes = list(np.atleast_1d(ax))
    fig.tight_layout()
    cf.set_axes_size_inches(fig, axes, w_in=w_in, h_in=cf.AXES_H_IN)
    size = cf.finish_points(axes[0], bar_points)
    fig.savefig(os.path.join(OUTDIR, f'{stem}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{stem}.png'), dpi=300,
                bbox_inches='tight')
    width = fig.get_size_inches()[0]
    print(f'  saved {stem} at {width:.2f} in wide, replicate dot s = {size}')


def write_sourcedata(rows, stem):
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTDIR, f'{stem}_sourcedata.csv'), index=False)


def intensity_rows(intensities, series_label_of, xcol):
    """Source-data rows for the right axis: per-run values plus the line points.

    `intensities` maps (method, category) to a `Run -> summed intensity` series,
    so the run names travel with the values and a reader can line the line up
    against the bars run by run.
    """
    rows = []
    for (method, cat), s in intensities.items():
        for rep, (run, v) in enumerate(s.items(), start=1):
            rows.append({'series': 'replicate point',
                         'metric': RIGHT_LABEL, 'method': series_label_of(method),
                         xcol: cat, 'run': run, 'replicate': rep,
                         'value': float(v)})
        rows.append({'series': 'line point (mean)', 'metric': RIGHT_LABEL,
                     'method': series_label_of(method), xcol: cat, 'run': '',
                     'replicate': np.nan, 'value': float(np.mean(s.to_numpy()))})
    return rows


def source_rows(summaries, series_label_of, cat_label, xcol):
    """Long-form source data: one row per replicate, one per bar height."""
    rows = []
    for (method, cat), s in summaries.items():
        for rep, (run, n) in enumerate(zip(s['runs'], s['per_run']), start=1):
            rows.append({'series': 'replicate point',
                         'metric': 'Peptides', 'method': series_label_of(method),
                         xcol: cat, 'run': run, 'replicate': rep, 'value': n,
                         'peptides_union': s['union'],
                         'peptides_with_cv': s['n_with_cv'],
                         'peptides_cv20': s['n_cv20'],
                         'median_cv_pct': round(s['median_cv_pct'], 2)})
        rows.append({'series': 'bar height (mean)', 'metric': 'Peptides',
                     'method': series_label_of(method), xcol: cat, 'run': '',
                     'replicate': np.nan,
                     'value': float(np.mean(s['per_run'])),
                     'peptides_union': s['union'],
                     'peptides_with_cv': s['n_with_cv'],
                     'peptides_cv20': s['n_cv20'],
                     'median_cv_pct': round(s['median_cv_pct'], 2)})
        rows.append({'series': 'bar height (CV < 20 %)', 'metric': 'Peptides',
                     'method': series_label_of(method), xcol: cat, 'run': '',
                     'replicate': np.nan, 'value': s['n_cv20'],
                     'peptides_union': s['union'],
                     'peptides_with_cv': s['n_with_cv'],
                     'peptides_cv20': s['n_cv20'],
                     'median_cv_pct': round(s['median_cv_pct'], 2)})
    return rows


def report(summaries, xlabel):
    """Print the table; the figure folder gets figures and source data only."""
    rows = []
    for (method, cat), s in summaries.items():
        rows.append({'method': method, xlabel: cat,
                     'peptides_mean': np.mean(s['per_run']),
                     'peptides_sd': np.std(s['per_run'], ddof=1),
                     'union': s['union'], 'with_cv': s['n_with_cv'],
                     'cv20': s['n_cv20'],
                     'cv20_pct_of_mean': 100 * s['n_cv20']
                     / np.mean(s['per_run']),
                     'median_cv_pct': s['median_cv_pct']})
    table = pd.DataFrame(rows)
    print(table.round(1).to_string(index=False))
    bad = table[table['cv20'] > table['peptides_mean']]
    if len(bad):
        print('WARNING: dark bar exceeds light bar (CV<20% set larger than the '
              'mean per-run count) in:')
        print(bad.to_string(index=False))
    return table


def report_intensity(intensities, xlabel):
    """Print the right-axis table, and say where each method's intensity peaks.

    The peak matters for the volume panel specifically: SAX SPEC's summed
    intensity is not flat across the dilution series even where its peptide count
    is, and which volume maximises it is the reason the default was chosen.
    """
    rows = [{'method': method, xlabel: cat,
             'intensity_mean': float(np.mean(s.to_numpy())),
             'intensity_sd': float(np.std(s.to_numpy(), ddof=1)),
             'n_runs': int(s.size)}
            for (method, cat), s in intensities.items()]
    table = pd.DataFrame(rows)
    print(f'\nsummed precursor intensity [{INTENSITY_SCALE:.0e}], '
          f'mean of the replicates:')
    print(table.round(3).to_string(index=False))
    peak = table.loc[table.groupby('method')['intensity_mean'].idxmax()]
    for _i, r in peak.iterrows():
        print(f'  {r["method"]:9s} peaks at {xlabel} = {r[xlabel]} '
              f'({r["intensity_mean"]:.3f})')
    return table


def cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


core.init_plotting()
