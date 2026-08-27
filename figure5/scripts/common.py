"""Shared configuration and helpers for the H032_E309 single-fiber analysis."""
from __future__ import annotations
import colorsys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spec_analytics as core
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
REPORT     = Path(_cfg.input_dir(__file__, 'report.parquet'))
DATA       = Path(_cfg.data_dir(__file__))
PLOTS      = Path(_cfg.output_dir(__file__))
SUPPORTING = Path(_cfg.output_dir(__file__, 'supporting'))
DF_FILTERED   = DATA / 'df_filtered.parquet'
SI_FILTERED   = DATA / 'sample_info_filtered.parquet'
SI_TYPED      = DATA / 'sample_info_typed.parquet'
TYPE_ORDER  = ['I', 'IIa', 'IIx', 'IIb']
TONE = 0.50
_TYPE_LIGHT = {'I': '#78B8E2', 'IIa': '#6BC88F', 'IIx': '#F5C842',
               'IIb': '#FA7A61', 'mixed': '#bdbdbd'}
_TYPE_DARK  = {'I': '#3274A1', 'IIa': '#3B8056', 'IIx': '#A58219',
               'IIb': '#CC683D', 'mixed': '#737373'}
def _blend(light, dark, t):
    a, b = np.array(mcolors.to_rgb(light)), np.array(mcolors.to_rgb(dark))
    return mcolors.to_hex(tuple((1 - t) * a + t * b))
def _undesaturate(color, prop=0.75):
    """Invert seaborn's palette desaturation.
    `plot_boxplot_with_points` goes through seaborn, which multiplies a palette's
    HLS saturation by `saturation=0.75` on the way in. Passing this pre-compensated
    value makes the drawn boxes land on TYPE_COLOR instead of being desaturated a
    second time, so that panel matches the raw-matplotlib panels beside it.
    """
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, min(1.0, s / prop)))
TYPE_COLOR = {k: _blend(_TYPE_LIGHT[k], _TYPE_DARK[k], TONE) for k in _TYPE_LIGHT}
TYPE_COLOR_BOX = {k: _undesaturate(v) for k, v in TYPE_COLOR.items()}
MUSCLE_ORDER = ['SOL', 'EDL']
MUSCLE_COLOR = {'SOL': core.PALETTE_SINGLE[2], 'EDL': core.PALETTE_SINGLE[0]}
SLOW_COLOR     = TYPE_COLOR['I']
FAST_COLOR     = TYPE_COLOR['IIb']
TEXTBOOK_COLOR = core.PALETTE_SINGLE[1]
GREY           = '#bdbdbd'
MYH_GENE = {'I': 'Myh7', 'IIa': 'Myh2', 'IIx': 'Myh1', 'IIb': 'Myh4'}
ACC_TO_MYH = {
    'Q91Z83': 'Myh7',
    'G3UW82': 'Myh2',
    'Q5SX40': 'Myh1',
    'Q5SX39': 'Myh4',
}
MYH_TO_TYPE = {'Myh7': 'I', 'Myh2': 'IIa', 'Myh1': 'IIx', 'Myh4': 'IIb'}
MYH_ORDER   = ['Myh7', 'Myh2', 'Myh1', 'Myh4']
EXCLUDE_GENES = {'Myh7', 'Myh2', 'Myh1', 'Myh4', 'Myh6', 'Myh8', 'Myh3', 'Myh13'}
TEXTBOOK_SLOW = {'Atp2a2', 'Myl2', 'Myl3', 'Casq2', 'Tnnc1', 'Tnni1', 'Tnnt1',
                 'Mb', 'Myoz2', 'Lmod2', 'Tpm3', 'Mybpc1', 'Cox7a1'}
TEXTBOOK_FAST = {'Atp2a1', 'Casq1', 'Tnnc2', 'Tnni2', 'Tnnt3', 'Pvalb',
                 'Mybpc2', 'Myl1', 'Myl11', 'Myoz1', 'Pdlim3', 'Actn3', 'Eno3'}
TEXTBOOK = TEXTBOOK_SLOW | TEXTBOOK_FAST
TEXTBOOK_NOTE = ('* = classical fiber-type marker '
                 '(Schiaffino & Reggiani 2011, Murgia 2015/2017/2021).')
MYHC_MARKER_SLOW = {'Myh7', 'Myh7b'}
MYHC_MARKER_FAST = {'Myh4'}
TEXTBOOK_DE_SLOW = TEXTBOOK_SLOW | MYHC_MARKER_SLOW
TEXTBOOK_DE_FAST = TEXTBOOK_FAST | MYHC_MARKER_FAST
TEXTBOOK_DE = TEXTBOOK_DE_SLOW | TEXTBOOK_DE_FAST
MIN_COVERAGE_OVERALL = 0.30
MIN_COVERAGE_GROUP   = 0.50
MIN_VALID_PER_ARM    = 3
SCORE_GROUPS         = ('I', 'IIa', 'IIb')
N_TOP                = 15
FC_THRESHOLD         = float(np.log2(1.5))
PADJ_THRESHOLD       = 0.05
def init(panel=(4, 4)):
    """House plotting defaults + make sure the output folders exist."""
    core.init_plotting()
    core.set_default_panel_size(*panel)
    for d in (DATA, PLOTS, SUPPORTING):
        d.mkdir(exist_ok=True)
def unbold(*axes):
    """House style bolds titles only; several core plots bold axis labels."""
    for ax in axes:
        ax.xaxis.label.set_fontweight('normal')
        ax.yaxis.label.set_fontweight('normal')
        for t in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            t.set_fontweight('normal')
def despine(*axes):
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
def lighten(color, factor=0.45):
    """Blend a colour towards white — the house 'secondary series' variant."""
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - ch) * (1 - factor) for ch in c)
def save_panel(fig, stem, sourcedata, *, out_dir=None, dpi=300):
    """Save one panel three ways and return the source-data frame.
    `sourcedata` holds the values actually drawn — pass a DataFrame, or a
    dict of {series_name: DataFrame} which is concatenated with a leading
    `series` column. It is never optional: every panel in this project shows
    measured data.
    Destination follows the stem: anything named `supporting_*` is not part of
    figure 5 and goes to supporting\\, everything else to the figure folder. That
    keeps the routing in one place instead of at every call site.
    """
    out_dir = Path(out_dir) if out_dir is not None else (
        SUPPORTING if stem.startswith('supporting') else PLOTS)
    if out_dir == PLOTS:
        for ax in fig.axes:
            title = ax.get_title()
            if title:
                print(f'  title -> caption: "{title}"')
                ax.set_title('')
    out_dir.mkdir(exist_ok=True)
    if isinstance(sourcedata, dict):
        frames = []
        for name, frame in sourcedata.items():
            frame = pd.DataFrame(frame).copy()
            frame.insert(0, 'series', name)
            frames.append(frame)
        sd = pd.concat(frames, ignore_index=True)
    else:
        sd = pd.DataFrame(sourcedata)
    fig.savefig(out_dir / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(out_dir / f'{stem}.png', dpi=dpi, bbox_inches='tight')
    sd.to_csv(out_dir / f'{stem}_sourcedata.csv', index=False)
    plt.close(fig)
    print(f'Wrote: {out_dir.name}/{stem}.pdf / .png / _sourcedata.csv')
    return sd
def load_filtered(columns=None):
    """Post-outlier-filter long precursor table (written by step 01)."""
    return pd.read_parquet(DF_FILTERED, columns=columns)
def load_typed():
    """sample_info with fiber-type calls (written by step 02)."""
    return pd.read_parquet(SI_TYPED)
def pg_log2_matrix(df):
    """(log2 pg_intensity matrix [protein_group x run], protein_group -> gene).
    One row per (run, protein_group); zeros are already NaN from the loader,
    and anything non-positive is dropped before the log.
    """
    df = df[df['genes'].notna() & (df['genes'].str.len() > 0)].copy()
    df['_gene'] = df['genes'].str.split(';').str[0]
    pg_wide = (df.drop_duplicates(['run', 'protein_group'])
                 .pivot(index='protein_group', columns='run',
                        values='pg_intensity'))
    pg_wide = np.log2(pg_wide.where(pg_wide > 0))
    pg_to_gene = (df.groupby('protein_group')['_gene']
                    .agg(lambda s: next((v for v in s if v), 'unannot')))
    pg_to_gene.name = 'gene'
    return pg_wide, pg_to_gene
def type_runs(sample_info, types=TYPE_ORDER):
    """{fiber_type: [run, ...]} for the pure-type fibers."""
    si = sample_info.set_index('run')
    return {t: si.index[si['fiber_type'] == t].tolist() for t in types}
def coverage_mask(pg_wide, sample_info):
    """The shared >=30%-overall AND >=50%-in-any-pure-group validity rule."""
    runs = type_runs(sample_info, SCORE_GROUPS)
    cov_total = pg_wide.count(axis=1) / pg_wide.shape[1]
    cov_group = pd.concat(
        {g: pg_wide[r].count(axis=1) / max(1, len(r)) for g, r in runs.items()},
        axis=1)
    return ((cov_total >= MIN_COVERAGE_OVERALL)
            & (cov_group.max(axis=1) >= MIN_COVERAGE_GROUP))
class MarkerSelection:
    """The slow-vs-fast marker selection: pure type I versus pure type IIb.
    **`score` is the log2 fold change, and only BH-significant proteins are
    ranked.** Both of those are round-2 corrections; the earlier version ranked on
    the difference of per-type mean z-scores, with the z taken across all 116
    fibers, and applied no significance requirement at all. Two problems with it,
    both measured rather than suspected:
    * **The top of that ranking was not reproducible.** The 15 slow scores spanned
      0.184 units (2.639 down to 2.455), so the set was a near-tie. Recomputing the
      z over the I+IIb subset only — a change any referee might ask for — replaced
      13 of the 15 slow markers and 9 of the 15 fast ones, even though the two
      scores correlate at Spearman 0.9975. Ranking on the log2 fold change removes
      the standardisation entirely and is far more stable: over 58 leave-one-fiber-
      out refits it retains 99.8 % of its membership (worst case 96.7 %), against
      94.6 % (worst 80.0 %) for the mean-z difference and 94.0 % (worst 80.0 %) for
      the Welch t.
    * **It named proteins the paper's own test cannot evaluate.** Pex1, Tmem233 and
      Tspan8 were among the 30 named while resting on 2, 2 and 1 observations in one
      arm, and all three are absent from the differential-expression table because
      that requires `MIN_VALID_PER_ARM` values in both arms. Applying the same
      requirement here means nothing is named that is not also tested.
    The Welch t was considered as the ranking statistic and rejected: it is no more
    stable than the old score, and with a handful of observations in one arm the
    within-group variance is underestimated, which inflates it (Ccdc167: type-I SD
    0.219 from 6 values against a 0.328 median across proteins, giving t = -17.3).
    A both-arms *coverage* threshold of 50 % was also considered and rejected. It
    would remove on/off proteins — present in most fast fibers, absent from most
    slow ones — which is a real biological pattern rather than a defect, so the gate
    here is a minimum count needed for the test, not a coverage fraction. Proteins
    whose signal is presence/absence are better served by a test on detection
    (Fisher exact on the 2x2 of detected/not) reported in the text than by an
    abundance ranking.
    IIa and IIx do not enter the contrast. The four-type resolution shows up in the
    composition and PCA panels instead.
    """
    def __init__(self, df, sample_info):
        from scipy import stats as _stats
        from statsmodels.stats.multitest import multipletests
        pg_wide, pg_to_gene = pg_log2_matrix(df)
        self.pg_to_gene = pg_to_gene
        self.n_pg_quantified = pg_wide.shape[0]
        self.n_fibers = pg_wide.shape[1]
        self.pg_wide_all = pg_wide
        self.pg_wide = pg_wide.loc[coverage_mask(pg_wide, sample_info)]
        self.pg_z = (self.pg_wide.sub(self.pg_wide.mean(axis=1), axis=0)
                                 .div(self.pg_wide.std(axis=1), axis=0))
        self.runs = type_runs(sample_info)
        self.mean_z = {t: self.pg_z[r].mean(axis=1) if r else
                          pd.Series(np.nan, index=self.pg_z.index)
                       for t, r in self.runs.items()}
        a, b = self.pg_wide[self.runs['I']], self.pg_wide[self.runs['IIb']]
        testable = ((a.notna().sum(axis=1) >= MIN_VALID_PER_ARM)
                    & (b.notna().sum(axis=1) >= MIN_VALID_PER_ARM)
                    & ~pg_to_gene.reindex(self.pg_wide.index)
                                 .isin(EXCLUDE_GENES).fillna(False))
        a, b = a[testable], b[testable]
        log2fc = a.mean(axis=1) - b.mean(axis=1)
        t_stat, p_raw = _stats.ttest_ind(a, b, axis=1, equal_var=False,
                                        nan_policy='omit')
        self.stats = pd.DataFrame({
            'gene': pg_to_gene.reindex(a.index),
            'n_I': a.notna().sum(axis=1), 'n_IIb': b.notna().sum(axis=1),
            'log2fc': log2fc,
            'welch_t': np.asarray(t_stat, dtype=float),
            'p': np.asarray(p_raw, dtype=float),
            'mean_z_diff': (self.mean_z['I'] - self.mean_z['IIb']).reindex(a.index),
        }, index=a.index)
        self.stats['padj'] = multipletests(self.stats['p'], method='fdr_bh')[1]
        self.stats['significant'] = (
            (self.stats['padj'] < PADJ_THRESHOLD)
            & (self.stats['log2fc'].abs() > FC_THRESHOLD))
        eligible = self.stats[self.stats['significant']]
        self.score = eligible['log2fc']
        self.top_slow = list(self.score.nlargest(N_TOP).index)
        self.top_fast = list(self.score.nsmallest(N_TOP).index)
    def gene(self, pg):
        return self.pg_to_gene.get(pg, '?')
    def label(self, pg, side):
        """Gene name, with '*' appended for canonical markers."""
        g = self.gene(pg)
        book = TEXTBOOK_SLOW if side == 'slow' else TEXTBOOK_FAST
        return f'{g} *' if g in book else g
    def pi_per_type(self, rows):
        """pi = log2FC x -log10(p) per fiber type (Xiao et al. 2014).
        log2FC = mean log2 intensity in type X minus the mean over all OTHER
        pure-type fibers; p from an uncorrected Welch t-test on the same split.
        """
        from scipy.stats import ttest_ind
        out = {}
        for t, runs_in in self.runs.items():
            runs_out = [r for k, rs in self.runs.items() if k != t for r in rs]
            m_in, m_out = self.pg_wide.loc[rows, runs_in], self.pg_wide.loc[rows, runs_out]
            log2fc = m_in.mean(axis=1) - m_out.mean(axis=1)
            pvals = []
            for pg in rows:
                a = m_in.loc[pg].dropna().values
                b = m_out.loc[pg].dropna().values
                if len(a) < 2 or len(b) < 2:
                    pvals.append(np.nan)
                    continue
                _, p = ttest_ind(a, b, equal_var=False)
                pvals.append(max(p, np.finfo(float).tiny))
            out[t] = log2fc * -np.log10(pd.Series(pvals, index=rows))
        return out
DE_COLUMNS = ['run', 'protein_group', 'genes', 'pg_intensity', 'engine']
DE_KWARGS = dict(
    level='protein',
    group_col='fiber_type',
    condition_a='I',
    condition_b='IIb',
    min_valid_per_condition=3,
    fc_threshold=FC_THRESHOLD,
    padj_threshold=PADJ_THRESHOLD,
)
def de_input(df, sample_info):
    """Long df restricted to the coverage-passing protein groups, ready for
    `core.plot_volcano(**common.DE_KWARGS)`."""
    pg_wide, _ = pg_log2_matrix(df)
    keep = pg_wide.index[coverage_mask(pg_wide, sample_info)]
    return df[df['protein_group'].isin(keep)]
def fiber_sort_order(sample_info):
    """Fibers ordered I -> IIa -> IIx -> IIb -> mixed, then slow->fast within.
    Keeps each pure-type block contiguous so the span labels above the
    composition strip and the heatmap are unambiguous.
    """
    si = sample_info.set_index('run')
    fr = si[['frac_I', 'frac_IIa', 'frac_IIx', 'frac_IIb']].rename(
        columns=lambda c: c.replace('frac_', ''))
    score = (1.00 * fr['I'].fillna(0) + 0.66 * fr['IIa'].fillna(0)
             + 0.33 * fr['IIx'].fillna(0))
    rank = {'I': 0, 'IIa': 1, 'IIx': 2, 'IIb': 3, 'mixed': 4, 'no_id': 5}
    order = (pd.DataFrame({'type_rank': si['fiber_type'].map(rank).fillna(5),
                           'score': score})
             .sort_values(['type_rank', 'score'], ascending=[True, False])
             .index.tolist())
    return order, fr
def type_spans(fiber_types, min_n=5):
    """Contiguous [(i0, i1, type)] runs of a fiber-type sequence."""
    bounds, prev = [], object()
    for i, t in enumerate(fiber_types):
        if t != prev:
            bounds.append((i, t))
            prev = t
    bounds.append((len(fiber_types), None))
    return [(i0, i1, t) for (i0, t), (i1, _) in zip(bounds, bounds[1:])
            if t is not None and i1 - i0 >= min_n]
def overlap_bars(ax, order, totals_by_cond, subset_by_cond, palette, *,
                 y_label, title, subset_label='CV<20%'):
    """House overlapping-bar panel: light bar = mean per-replicate total,
    dark bar of the same hue = the group-level subset, individual replicate
    totals scattered on top.
    `core.plot_overlapping_bars` does the same thing but builds its own
    figure; this variant draws onto a supplied axis so it can sit in a
    multi-panel row.
    """
    x = np.arange(len(order))
    max_y = max((max(totals_by_cond[c]) for c in order), default=1)
    rng = np.random.default_rng(0)
    rows = []
    for j, cond in enumerate(order):
        color = palette[cond]
        totals = np.asarray(totals_by_cond[cond], dtype=float)
        higher, lower = float(totals.mean()), float(subset_by_cond[cond])
        ax.bar(x[j], higher, width=0.7, color=lighten(color, 0.45),
               edgecolor='darkgray', linewidth=1.0)
        ax.bar(x[j], lower, width=0.7, color=color,
               edgecolor='black', linewidth=1.0)
        if totals.size > 1:
            jitter = rng.uniform(-0.13, 0.13, totals.size)
            ax.scatter(x[j] + jitter, totals, c='black', s=18, alpha=0.75,
                       zorder=5, edgecolors='white', linewidths=0.4)
        ax.text(x[j], higher + max_y * 0.012, f'{int(higher):,}',
                ha='center', va='bottom', fontsize=9)
        ax.text(x[j], lower - max_y * 0.012, f'{int(lower):,}\n{subset_label}',
                ha='center', va='top', fontsize=8,
                color='white' if lower > max_y * 0.12 else 'black')
        rows.append({'condition': cond, 'mean_total': higher,
                     'subset_count': lower,
                     'replicate_totals': ';'.join(str(int(v)) for v in totals)})
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max_y * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.tick_params(axis='both', labelsize=10)
    despine(ax)
    return pd.DataFrame(rows)
def annotate_in_margins(ax, right_hits, left_hits, *, color='black',
                        fontsize=7.5, expand=0.26, fontweight='bold',
                        leader_lw=0.4, leader_alpha=0.55):
    """Label points in the left/right margin with leader lines to the dot.
    `right_hits` / `left_hits` are [(label, x, y), ...]. Labels sit outside
    the data envelope so the leader line never has to cross the dense cluster;
    each label stays as close to its own dot as the row spacing allows
    (greedy top-down placement), which keeps the lines short and readable.
    x-limits are expanded first so the labels stay inside the visible axes.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]
    data_lo = min([x for _, x, _ in right_hits + left_hits], default=xlim[0])
    data_hi = max([x for _, x, _ in right_hits + left_hits], default=xlim[1])
    ax.set_xlim(xlim[0] - (x_span * expand if left_hits else 0),
                xlim[1] + (x_span * expand if right_hits else 0))
    pad = y_span * 0.035
    y_top, y_bot = ylim[1] - pad, ylim[0] + pad
    ax_height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    row_gap = fontsize * 1.45 / ax_height_pt * y_span
    for hits, side in ((right_hits, 'right'), (left_hits, 'left')):
        if not hits:
            continue
        hits = sorted(hits, key=lambda h: -h[2])
        min_gap = min((y_top - y_bot) / len(hits), row_gap)
        margin = x_span * 0.09
        x_label = data_hi + margin if side == 'right' else data_lo - margin
        ha = 'left' if side == 'right' else 'right'
        prev = y_top + min_gap
        n = len(hits)
        for i, (label, x_data, y_data) in enumerate(hits):
            floor = y_bot + (n - 1 - i) * min_gap
            y_lab = max(min(y_data, prev - min_gap), floor)
            prev = y_lab
            ax.annotate(label, xy=(x_data, y_data), xytext=(x_label, y_lab),
                        fontsize=fontsize, color=color, fontweight=fontweight,
                        ha=ha, va='center',
                        arrowprops=dict(arrowstyle='-', color=color,
                                        lw=leader_lw, alpha=leader_alpha,
                                        shrinkA=1, shrinkB=2))
