"""Shared geometry and the dual-axis grouped-bar construction for figure 2."""
import numpy as np
import pyarrow.parquet as pq
import matplotlib.colors as mcolors
import spec_analytics as core
QVALUE = 0.01
MIN_VALUES_FOR_CV = 3
CV_THRESHOLD = 0.20
FONTSIZE = 10
LEGEND_FONTSIZE = 9.5
AXES_H_IN = 2.50
set_axes_size_inches = core.set_axes_size_inches
PANEL_W_IN = 4.20
POINT_SIZE_PLACEHOLDER = 22
BAR_FRAC = 0.88
GROUP_GAP = 0.8
EDGE_PAD = 0.5
LEGEND_DARK = '#595959'
def lighten(color, factor=0.45):
    """Blend `color` toward white — the house light/dark bar variant."""
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - ch) * (1 - factor) for ch in c)
def bar_width_inches(fig, ax, n_bar_series, n_groups):
    """Drawn bar width in inches, for reporting how dense a panel really is."""
    fig.canvas.draw()
    axes_in = ax.get_position().width * fig.get_size_inches()[0]
    x0, x1 = ax.get_xlim()
    return BAR_FRAC * axes_in / (x1 - x0)
def draw_grouped_dual(ax_left, ax_right, categories, series, *,
                      point_size=POINT_SIZE_PLACEHOLDER, seed=0,
                      line_width=1.6, marker_size=5.0):
    """Grouped bars on the left axis, line plots on the right axis.
    `series` is a list of dicts:
        label   legend text
        method  method identity, used to align a right-axis line with the bars
                of the same method
        color   method hue
        axis    'left' (bars) or 'right' (line)
        values  {category: sequence of per-replicate values}
    **Only the left-axis series occupy bar slots**, so adding the second metric
    costs no bar width: b, c, d and e come out at 16, 15, 15 and 6 bars rather
    than 32, 30, 30 and 12, which is what lets the bars stay wide enough to carry
    the larger replicate marker. Drawing the right-axis metric as a line rather
    than a bar also survives a wide y range — a marker at 1% of the axis is still
    a marker, where a bar 1% tall is invisible. That is what makes panel d
    readable at all, its summed quantity spanning 90x.
    Each right-axis line runs through the x centres of its own method's bars, so
    it sits above the bars it belongs to rather than at the group centre.
    Bars and line markers are both drawn at the mean of the replicate values with
    the individual replicates overlaid, so a plotted height is always the mean of
    the points on it rather than a ratio of sums.
    Returns (heights, bar_point_collections): the list of
    (series_label, category, mean) triples for the source data, and the
    replicate-dot collections drawn on the bars, which `finish_points` resizes
    once the layout is final.
    """
    rng = np.random.default_rng(seed)
    left = [s for s in series if s['axis'] == 'left']
    right = [s for s in series if s['axis'] == 'right']
    n = len(left)
    step = n + GROUP_GAP
    jitter = 0.22 * BAR_FRAC
    heights, bar_points = [], []
    slot = {}
    for k, s in enumerate(left):
        for i, cat in enumerate(categories):
            vals = np.asarray(s['values'][cat], dtype=float)
            centre = i * step + k + 0.5
            slot[(s['method'], cat)] = centre
            mean = float(np.mean(vals))
            ax_left.bar(centre, mean, BAR_FRAC, color=lighten(s['color']),
                        edgecolor=s['color'], linewidth=0.7, zorder=2)
            bar_points.append(ax_left.scatter(
                centre + rng.uniform(-jitter, jitter, size=vals.size), vals,
                s=point_size, color='black', alpha=0.85, linewidth=0.3,
                edgecolor='white', zorder=5))
            heights.append((s['label'], cat, mean))
    centres = [i * step + n / 2 for i in range(len(categories))]
    for s in right:
        xs = [slot.get((s['method'], cat), centres[i])
              for i, cat in enumerate(categories)]
        means = [float(np.mean(np.asarray(s['values'][cat], dtype=float)))
                 for cat in categories]
        for x, cat in zip(xs, categories):
            vals = np.asarray(s['values'][cat], dtype=float)
            if vals.size > 1:
                ax_right.scatter(np.full(vals.size, x), vals, s=0.5 * point_size,
                                 color=s['color'], alpha=0.6, linewidth=0.0,
                                 zorder=6)
        ax_right.plot(xs, means, color=s['color'], linewidth=line_width,
                      marker='o', markersize=marker_size,
                      markerfacecolor=s['color'], markeredgecolor='white',
                      markeredgewidth=0.7, zorder=7)
        heights.extend((s['label'], cat, m) for cat, m in zip(categories, means))
    first_edge = 0.5 - BAR_FRAC / 2
    last_edge = (len(categories) - 1) * step + n - 0.5 + BAR_FRAC / 2
    ax_left.set_xlim(first_edge - EDGE_PAD, last_edge + EDGE_PAD)
    ax_left.set_xticks(centres)
    ax_left.set_xticklabels([str(c) for c in categories], fontsize=FONTSIZE)
    ax_right.tick_params(axis='x', length=0)
    return heights, bar_points
def style_dual_axes(ax_left, ax_right, *, left_label, right_label,
                    left_max, right_max, headroom=1.14):
    """Zero-based twin axes, house tick sizes, no top spine on either axis."""
    ax_left.set_ylim(0, left_max * headroom)
    ax_right.set_ylim(0, right_max * headroom)
    ax_left.set_ylabel(left_label, fontsize=FONTSIZE)
    ax_right.set_ylabel(right_label, fontsize=FONTSIZE)
    for ax in (ax_left, ax_right):
        ax.tick_params(labelsize=FONTSIZE)
        ax.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_visible(True)
    ax_right.spines['right'].set_linewidth(
        ax_left.spines['left'].get_linewidth())
def dual_legends(ax, method_handles, metric_handles, *, row_gap=0.075):
    """Two frameless keys stacked above the axes: metrics, then methods on top.
    The bars encode two orthogonal things — hue is the method, alpha is the
    metric — so one combined legend would make the reader decode both from a
    single list. Two keys read as what they are. They are **stacked rather than
    placed left and right**: side by side, the two key rows together run wider
    than a half-row panel's axes and the labels overlap.
    """
    kw = dict(frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=0.8,
              handleheight=0.9, handletextpad=0.35, borderpad=0.0,
              columnspacing=0.8)
    metrics = ax.legend(handles=metric_handles, loc='lower left',
                        bbox_to_anchor=(0.0, 1.0), ncol=len(metric_handles),
                        **kw)
    if not method_handles:
        return
    ax.add_artist(metrics)
    ax.legend(handles=method_handles, loc='lower left',
              bbox_to_anchor=(0.0, 1.0 + row_gap),
              ncol=len(method_handles), **kw)
def metric_key(left_label, right_label):
    """Grey bar / grey line pair naming the two metrics and their axes.
    Mark type, not alpha, now separates the metrics: a bar is the left axis and a
    line is the right one. Drawn in grey so the key states the mark rather than a
    method — hue is reserved for method identity, per the house rule for a
    multi-item readout.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    return [Patch(facecolor=lighten(LEGEND_DARK), edgecolor=LEGEND_DARK,
                  linewidth=0.7, label=left_label),
            Line2D([0], [0], color=LEGEND_DARK, linewidth=1.6, marker='o',
                   markersize=5.0, markerfacecolor=LEGEND_DARK,
                   markeredgecolor='white', markeredgewidth=0.7,
                   label=right_label)]
def method_key(labels_colors):
    from matplotlib.patches import Patch
    return [Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=l)
            for l, c in labels_colors]
def finish_points(ax_left, bar_points):
    """Resize the bar replicate dots to the finished bar width.
    Call after `set_axes_size_inches`, i.e. once nothing will move again. Only the
    bar dots are passed — the right-axis line's own replicate dots are
    deliberately smaller and hue-coloured, so they must not be swept up by a
    blanket resize of every collection on the axes.
    """
    return core.scale_replicate_points(ax_left, bar_points)
PEPTIDE_COLUMN = 'Modified.Sequence'
QUANTITY_COLUMN = 'Precursor.Normalised'
RAW_QUANTITY_COLUMN = 'Precursor.Quantity'
def read_peptides(path, *, runs=None, quantity_column=None):
    """Peptide-level quantity per run from one DIA-NN combined report.
    A **peptide is a peptidoform** — a `Modified.Sequence`, i.e. sequence plus
    localised modifications — and its quantity in a run is the sum of its
    precursors' `QUANTITY_COLUMN` (DIA-NN's normalised quantity by default; see
    the note above the constant), so charge states are summed and nothing else
    is. Pass `quantity_column=RAW_QUANTITY_COLUMN` for the panels that measure
    absolute recovery, where normalising away the run's scale would remove the
    quantity being measured. This is the definition `spec_analytics` uses for `peptide_id`
    and `peptide_intensity` (`io/diann.py`), and the one Supplementary Figure 1's
    overlap panel already counts, so the whole paper says "peptide" for one thing.
    It was `Stripped.Sequence` until 2026-08-09, which additionally merged the
    modified and unmodified forms of a sequence and read ~12 % low on the SPEC and
    PAC arms of E305 but only ~0.6 % low on ISD+, where a sequence is rarely seen
    in both forms — so the change is not a uniform rescaling and every peptide
    number in the manuscript moved with it.
    Two roll-ups deliberately stay on the stripped sequence and must not be
    switched: `supplement_lc_hydrophobicity`, because `core.gravy` needs a bare
    amino-acid string, and figure 3's coverage panel, because peptides there are
    mapped onto protein sequences.
    Filter is the paper's: `Q.Value <= 0.01`, `PG.Q.Value <= 0.01`, `Decoy == 0`
    where the column exists, and `Precursor.Quantity > 0` so a zero is never
    averaged in as a measurement. Only the protein-group q-value actually removes
    rows — DIA-NN already writes the report at 1 % precursor FDR with no decoys —
    but the rest are kept so the filter does not depend on that staying true.
    Pushed down to pyarrow at read time. E256 is 9.0 M rows and materialising it
    in pandas before filtering is what makes the naive version run out of memory.
    """
    quantity_column = quantity_column or QUANTITY_COLUMN
    available = set(pq.ParquetFile(path).schema_arrow.names)
    if PEPTIDE_COLUMN not in available:
        raise ValueError(f'{path}: no {PEPTIDE_COLUMN} column, so peptides cannot '
                         f'be counted at the peptidoform level')
    if quantity_column not in available:
        raise ValueError(f'{path}: no {quantity_column} column')
    columns = [c for c in ('Run', PEPTIDE_COLUMN, quantity_column,
                           'Q.Value', 'PG.Q.Value', 'Decoy') if c in available]
    filters = [('Q.Value', '<=', QVALUE), ('PG.Q.Value', '<=', QVALUE),
               (RAW_QUANTITY_COLUMN, '>', 0)]
    if 'Decoy' in available:
        filters.append(('Decoy', '==', 0))
    if RAW_QUANTITY_COLUMN not in columns:
        columns.append(RAW_QUANTITY_COLUMN)
    d = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    if runs is not None:
        d = d[d['Run'].isin(runs)]
    return (d.groupby(['Run', PEPTIDE_COLUMN], sort=False,
                      observed=True)[quantity_column]
            .sum().rename('quantity').reset_index()
            .rename(columns={PEPTIDE_COLUMN: 'peptide'}))
def counts_and_cv(pep):
    """(per-run peptide counts, CV per peptide) for one condition's replicates.
    CV is **SD / mean on linear quantities** with `ddof=1`, over the observed
    values only, on whatever quantity `read_peptides` returned — normalised by
    default, so a run's overall recovery does not enter — missing peptides are skipped, never imputed as zero, which is
    the definition used everywhere else in the paper. Peptides with fewer than
    `MIN_VALUES_FOR_CV` observed values have no CV and are excluded from the dark
    bar but still counted in the light one.
    """
    wide = pep.pivot_table(index='peptide', columns='Run', values='quantity')
    per_run = wide.notna().sum(axis=0)
    enough = wide[wide.notna().sum(axis=1) >= MIN_VALUES_FOR_CV]
    cv = enough.std(axis=1, ddof=1) / enough.mean(axis=1)
    return per_run, cv
def protein_group_cv(path, *, runs=None):
    """CV per protein group across one condition's replicates, on PG.MaxLFQ.
    This is what the dark bar of supplementary figure 3's d-g counts. Kept
    separate from the per-run protein-group COUNT each panel already computes,
    deliberately: the counts are `Protein.Group` nunique under the
    `PG.Q.Value` filter and every protein-group number in the manuscript comes
    from them, so they are not re-derived here and cannot move.
    **PG.MaxLFQ, matching figure 4's protein-group CVs and the methods statement
    that protein quantities are MaxLFQ values.** One value per (run, protein
    group), so the frame is deduplicated rather than summed. CV is SD / mean on
    linear values with `ddof=1` over the observed values only, missing values
    skipped and never imputed as zero, for groups with at least
    `MIN_VALUES_FOR_CV` observations — the paper's definition throughout.
    The MaxLFQ universe is 0.1-0.15 % smaller than the count universe (8,371 of
    8,380 groups in E306 SPEC at 5 uL), because a group can pass the q-value
    filter in a run without receiving a MaxLFQ value. That is far below anything
    readable on the panel and is dominated anyway by the `MIN_VALUES_FOR_CV`
    requirement, which is what actually decides the dark bar.
    """
    available = set(pq.ParquetFile(path).schema_arrow.names)
    if 'PG.MaxLFQ' not in available:
        raise ValueError(f'{path}: no PG.MaxLFQ column, so protein-group CVs '
                         f'cannot be computed')
    columns = [c for c in ('Run', 'Protein.Group', 'PG.MaxLFQ', 'Q.Value',
                           'PG.Q.Value', 'Decoy') if c in available]
    filters = [('Q.Value', '<=', QVALUE), ('PG.Q.Value', '<=', QVALUE),
               (RAW_QUANTITY_COLUMN, '>', 0)]
    if 'Decoy' in available:
        filters.append(('Decoy', '==', 0))
    d = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    if runs is not None:
        d = d[d['Run'].isin(runs)]
    d = d[d['PG.MaxLFQ'].notna() & (d['PG.MaxLFQ'] > 0)]
    d = d.drop_duplicates(['Run', 'Protein.Group'])
    wide = d.pivot(index='Protein.Group', columns='Run', values='PG.MaxLFQ')
    enough = wide[wide.notna().sum(axis=1) >= MIN_VALUES_FOR_CV]
    return enough.std(axis=1, ddof=1) / enough.mean(axis=1)
def n_cv20(cv):
    """Entities below the CV threshold — the dark bar's height."""
    return int((cv < CV_THRESHOLD).sum())
def summarise(pep):
    """One condition -> dict of the plotted quantities plus context numbers."""
    per_run, cv = counts_and_cv(pep)
    return {
        'per_run': [int(v) for v in per_run.sort_index().to_numpy()],
        'runs': list(per_run.sort_index().index),
        'union': int(pep['peptide'].nunique()),
        'n_with_cv': int(cv.size),
        'n_cv20': int((cv < CV_THRESHOLD).sum()),
        'median_cv_pct': float(100 * cv.median()) if cv.size else np.nan,
    }
