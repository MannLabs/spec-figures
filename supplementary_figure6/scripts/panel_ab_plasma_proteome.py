"""Supplementary figure 6a,b — the unfractionated plasma proteome (H032_E237)."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
# One search per preparation; the old combined 8-run report stays in input/ as the
# record of what round 2 reported.
REPORTS = {'ISD': _cfg.input_dir(__file__, 'H032_E237', 'ISD',
                               'report.parquet'),
           'SAX SPEC': _cfg.input_dir(__file__, 'H032_E237', 'SAX_SPEC',
                                    'report.parquet')}
OUTDIR = _cfg.output_dir(__file__)

ROW_CONDITION = {'A': 'ISD', 'C': 'SAX SPEC'}
# Order and hues locked to figure 6, whose glyco panels this supplements: SAX SPEC
# keeps its coral, the in-solution digest takes figure 2's ISD lavender.
CONDITIONS = ['ISD', 'SAX SPEC']
COLOR = {'ISD': core.PALETTE_SINGLE[1], 'SAX SPEC': core.PALETTE_SINGLE[0]}

QVALUE = 0.01
N_REPLICATES = 4
FONTSIZE = 8               # figure 6's size; these panels are placed at 1:1
BAR_WIDTH_IN = 0.38
SHOW_VALUE_LABELS = False  # as everywhere else; the numbers are in the source data
POINT_SIZE = core.replicate_point_size(BAR_WIDTH_IN)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
frames = []
for condition, path in REPORTS.items():
    x = pd.read_parquet(path, columns=[
        'Run', 'Protein.Group', 'PG.MaxLFQ', 'Precursor.Id',
        'Precursor.Quantity', 'Q.Value', 'PG.Q.Value', 'Decoy'])
    x = x[(x['Decoy'] == 0) & (x['Q.Value'] < QVALUE)
          & (x['PG.Q.Value'] < QVALUE) & (x['Precursor.Quantity'] > 0)]
    frames.append(x.assign(condition=condition))
raw = pd.concat(frames, ignore_index=True)

_long = pd.DataFrame({
    'run': raw['Run'].astype(str),
    'protein_group': raw['Protein.Group'].astype(str),
    'precursor_id': raw['Precursor.Id'].astype(str),
    'precursor_intensity': raw['Precursor.Quantity'].astype(float)})
_long['pg_intensity'] = core.compute_directlfq_pg_intensity(_long, num_cores=1)
_long['pg_intensity'] = _long['pg_intensity'].replace(0, np.nan)
raw['PG.MaxLFQ'] = _long['pg_intensity'].to_numpy()

per_run = (raw.groupby(['condition', 'Run'])['Protein.Group'].nunique()
           .rename('protein_groups').reset_index())
counts = per_run.groupby('condition').size()
if not (counts == N_REPLICATES).all():
    raise ValueError(f'expected {N_REPLICATES} runs per condition: '
                     f'{counts.to_dict()}')

summary = per_run.groupby('condition')['protein_groups'].agg(['mean', 'std'])
print('protein groups per run:')
print(per_run.to_string(index=False))
print('\nmean +/- SD over 4 replicates:')
print(summary.reindex(CONDITIONS).round(1).to_string())
welch = stats.ttest_ind(
    per_run.loc[per_run['condition'] == 'ISD', 'protein_groups'],
    per_run.loc[per_run['condition'] == 'SAX SPEC', 'protein_groups'],
    equal_var=False)
print(f'Welch t-test, ISD vs SAX SPEC: p = {welch.pvalue:.3f} '
      f'(difference '
      f'{summary.loc["SAX SPEC", "mean"] - summary.loc["ISD", "mean"]:+.1f} '
      f'protein groups)')

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    'run': raw['Run'].astype(str),
    'engine': 'diann',
    'protein_group': raw['Protein.Group'].astype(str),
    'pg_intensity': raw['PG.MaxLFQ'].astype(float).replace(0, np.nan),
}).drop_duplicates(['run', 'protein_group'])

sample_info = (raw[['Run', 'condition']].drop_duplicates()
               .rename(columns={'Run': 'run', 'condition': 'condition2'}))
sample_info['engine'] = 'diann'
# Without this every point label would read "1" — `load_experiments` would assign
# replicate = 1 to all runs, and the plate-well suffix is the real replicate index.
sample_info['replicate'] = sample_info['run'].str[-1].astype(int)

# Detection overlap, for the record. Not plotted: 134 + 94 of 1,630 is a 86 % core,
# which a Venn would spend a whole panel to say.
detected = (df.dropna(subset=['pg_intensity'])
            .merge(sample_info[['run', 'condition2']], on='run'))
sets = {c: set(g['protein_group'])
        for c, g in detected.groupby('condition2')}
union = sets['ISD'] | sets['SAX SPEC']
print(f'\nprotein groups detected: union {len(union):,}; '
      f'{len(sets["ISD"] - sets["SAX SPEC"])} only ISD, '
      f'{len(sets["SAX SPEC"] - sets["ISD"])} only SAX SPEC, '
      f'{len(sets["ISD"] & sets["SAX SPEC"]):,} in both')

# ---------------------------------------------------------------------------
# a — protein groups per run
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(2.2, 2.6))
for i, condition in enumerate(CONDITIONS):
    vals = per_run.loc[per_run['condition'] == condition,
                       'protein_groups'].to_numpy()
    ax.bar(i, vals.mean(), 0.6, color=COLOR[condition], edgecolor='black',
           linewidth=0.5, zorder=2)
    ax.scatter(np.full(len(vals), i) + rng.uniform(-0.09, 0.09, len(vals)), vals,
               s=POINT_SIZE, color='black', alpha=0.75, linewidth=0.3,
               edgecolor='white', zorder=5)
    if SHOW_VALUE_LABELS:
        ax.text(i, vals.max() * 1.03, f'{vals.mean():,.0f}', ha='center',
                va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels(CONDITIONS, fontsize=FONTSIZE, rotation=45, ha='right')
ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
# Zero-based, because the claim is that the two are the same. A zoomed axis would
# manufacture a difference out of a 1.7 % gap that a Welch test does not support.
ax.set_ylim(0, per_run['protein_groups'].max() * 1.18)
ax.set_ylabel('Protein groups', fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.canvas.draw()          # fixed drawn bar width, as in figures 2-4 and 6
ax_w = ax.get_position().width * fig.get_size_inches()[0]
w_data = BAR_WIDTH_IN * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax_w
for patch in ax.patches:
    centre = patch.get_x() + patch.get_width() / 2
    patch.set_width(w_data)
    patch.set_x(centre - w_data / 2)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'panel_a_plasma_protein_groups.pdf'),
            bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_a_plasma_protein_groups.png'), dpi=300,
            bbox_inches='tight')

pd.concat([
    per_run.assign(series='replicate point')[
        ['series', 'condition', 'Run', 'protein_groups']],
    pd.DataFrame([{'series': 'bar height (mean)', 'condition': c, 'Run': '',
                   'protein_groups': summary.loc[c, 'mean']}
                  for c in CONDITIONS])], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'panel_a_plasma_protein_groups_sourcedata.csv'),
    index=False)

# ---------------------------------------------------------------------------
# b — quantity correlation, styled as figure 4d/4e
# ---------------------------------------------------------------------------
def restyle(ax, fig, filename, *, xlabel, ylabel, rasterize_points=False):
    """Figure 4's `restyle`, verbatim in behaviour.

    `core` bolds axis labels and sets its r / rho / n box at 10 pt; this figure, like
    figure 4, wants normal weight at `FONTSIZE` and a 7 pt annotation on a white
    patch.

    `rasterize_points` flattens the point cloud into a single 600 dpi image inside
    the PDF, as figure 4d and 4e do. It is on here for consistency with them rather
    than out of necessity — 1,402 individually coloured paths would not trouble
    Illustrator the way their ~7,000 do. **Rasterisation is surgical:** only the
    collections are flattened, so axes, ticks, both labels and the r / rho / n box
    stay vector and editable, and the image lands inside the axes rectangle rather
    than over the whole figure.
    """
    for axis_label in (ax.xaxis.label, ax.yaxis.label):
        axis_label.set_fontweight('normal')
        axis_label.set_fontsize(FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    for text in ax.texts:
        text.set_fontsize(7)
        if text.get_bbox_patch() is not None:
            text.set_bbox(dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.set_title('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if rasterize_points:
        for collection in ax.collections:
            collection.set_rasterized(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{filename}.pdf'), bbox_inches='tight',
                dpi=600 if rasterize_points else None)
    fig.savefig(os.path.join(OUTDIR, f'{filename}.png'), dpi=300,
                bbox_inches='tight')


fig, ax, plot_df = core.plot_correlation(
    df, sample_info, level='protein', group_col='condition2',
    condition_a='ISD', condition_b='SAX SPEC',
    figsize=(2.2, 2.5), point_size=4, alpha=0.7, color_by_density=True,
    show_diagonal=True)
restyle(ax, fig, 'panel_b_plasma_correlation', rasterize_points=True,
        xlabel=r'$\log_2$ intensity, ISD',
        ylabel=r'$\log_2$ intensity, SAX SPEC')

col_isd, col_spec = 'log2_mean_ISD', 'log2_mean_SAX SPEC'
resid = plot_df[col_spec] - plot_df[col_isd]
rho = stats.spearmanr(plot_df[col_isd], plot_df[col_spec]).statistic
pearson = stats.pearsonr(plot_df[col_isd], plot_df[col_spec]).statistic
print(f'\npanel b: n = {len(plot_df):,} protein groups quantified in both')
print(f'  Spearman rho = {rho:.4f}, Pearson r on log2 = {pearson:.4f}')
print(f'  log2 ratio SAX SPEC / ISD: median {resid.median():+.3f}, '
      f'IQR {resid.quantile(.25):+.3f} to {resid.quantile(.75):+.3f}')

plot_df.assign(log2_ratio_SPEC_over_ISD=resid).round(4).to_csv(
    os.path.join(OUTDIR, 'panel_b_plasma_correlation_sourcedata.csv'), index=False)

print(f'\nSaved panels a and b to {OUTDIR}')
