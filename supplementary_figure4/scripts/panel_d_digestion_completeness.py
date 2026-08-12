"""Supplementary figure 3d — digestion completeness of the FFPE preparations."""

import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyarrow.parquet as pq

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
# Read from figure 4's input tree rather than keeping a second 400 MB copy here,
# the same arrangement supplementary figure 3 uses for figure 2's experiments.
FIG4_INPUT = _cfg.cross_input('figure4')
FFPE_REPORTS = {
    'Single-shot SPEC': os.path.join(FIG4_INPUT, 'H032_E127',
                                     'single-shot_SPEC', 'report.parquet'),
    'Single-shot ISD+': os.path.join(FIG4_INPUT, 'H032_E127',
                                     'single-shot_ISD+', 'report.parquet'),
    'Bulk ISD+': os.path.join(FIG4_INPUT, 'H032_E127',
                              'bulk_ISD+', 'report.parquet')}
ORGAN_REPORTS = {
    'SPEC': os.path.join(FIG4_INPUT, 'H032_E170', 'SPEC', 'report.parquet'),
    'PAC': os.path.join(FIG4_INPUT, 'H032_E170', 'PAC', 'report.parquet')}
OUTDIR = _cfg.output_dir(__file__)

QVALUE = 0.01
BATCH_ROWS = 500_000
FONTSIZE = 8
STEM = 'panel_d_digestion_completeness'

# Same maps, order, hues and dodge as panels a-c, so a condition keeps its identity.
ROW_CONDITION = {'A': 'Single-shot SPEC', 'B': 'Single-shot ISD+',
                 'C': 'Bulk ISD+'}
ORDER = ['Single-shot ISD+', 'Bulk ISD+', 'Single-shot SPEC']
XTICK = {'Single-shot ISD+': 'Single-shot\nISD+', 'Bulk ISD+': 'Bulk\nISD+',
         'Single-shot SPEC': 'Single-shot\nSPEC'}
ROW_ORGAN = {'A': 'Liver', 'B': 'Brain', 'C': 'Heart', 'D': 'Kidney',
             'E': 'Testis', 'F': 'Lung'}
ORGAN_ORDER = ['Testis', 'Kidney', 'Brain', 'Lung', 'Liver', 'Heart']
METHODS = ['PAC', 'SPEC']
DODGE = {'PAC': -0.16, 'SPEC': 0.16}


def lighten(color, factor=0.45):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * factor)


COLOR = {'Single-shot SPEC': core.PALETTE_SINGLE[0],
         'Bulk ISD+': core.PALETTE_SINGLE[5],
         'Single-shot ISD+': lighten(core.PALETTE_SINGLE[5]),
         'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2]}

POINT_SIZE = 18
YLIM = (50, 100)
YTICKS = [50, 60, 70, 80, 90, 100]
YLABEL = 'Fully cleaved precursors [%]'

_mc_memo = {}


def mc0_per_run(path):
    """Per-run count- and intensity-weighted fully-cleaved rate.

    Streamed in batches: E170 is 5.7 M rows, and the missed-cleavage count has to
    touch every distinct stripped sequence.
    """
    columns = ['Run', 'Stripped.Sequence', 'Precursor.Quantity', 'Q.Value',
               'PG.Q.Value']
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [c for c in columns if c in available]
    acc = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS,
                                                   columns=columns):
        d = batch.to_pandas()
        d = d[(d['Q.Value'] <= QVALUE) & (d['PG.Q.Value'] <= QVALUE)
              & (d['Precursor.Quantity'] > 0)]
        if d.empty:
            continue
        for seq in d['Stripped.Sequence'].unique():
            if seq not in _mc_memo:
                _mc_memo[seq] = core.count_missed_cleavages(seq,
                                                            protease='trypsin')
        d = d.assign(is_mc0=d['Stripped.Sequence'].map(_mc_memo) == 0)
        for run, g in d.groupby('Run', sort=False):
            a = acc.setdefault(run, [0, 0, 0.0, 0.0])
            a[0] += len(g)
            a[1] += int(g['is_mc0'].sum())
            a[2] += float(g['Precursor.Quantity'].sum())
            a[3] += float(g.loc[g['is_mc0'], 'Precursor.Quantity'].sum())
    return pd.DataFrame([
        {'run': run, 'precursors': n, 'mc0_by_count': 100 * n0 / n,
         'mc0_by_intensity': 100 * q0 / q}
        for run, (n, n0, q, q0) in acc.items()])


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
ffpe = pd.concat([mc0_per_run(path).assign(condition=cond)
                  for cond, path in FFPE_REPORTS.items()], ignore_index=True)
ffpe['tag'] = ffpe['run'].str.extract(r'_([A-C]\d{1,2})$')[0]

organs = pd.concat([mc0_per_run(path).assign(method=method)
                    for method, path in ORGAN_REPORTS.items()], ignore_index=True)
organs['tag'] = organs['run'].str.extract(r'_([A-G]\d{1,2})$')[0]
organs['organ'] = organs['tag'].str[0].map(ROW_ORGAN)
if organs['organ'].isna().any():
    raise ValueError('runs whose plate row maps to no organ')

print('FFPE preparations, fully cleaved [%] (mean, SD, n):')
print(ffpe.groupby('condition')[['mc0_by_count', 'mc0_by_intensity']]
      .agg(['mean', 'std', 'size']).reindex(ORDER).round(2).to_string())
spec = ffpe.loc[ffpe['condition'] == 'Single-shot SPEC', 'mc0_by_count'].mean()
bulk = ffpe.loc[ffpe['condition'] == 'Bulk ISD+', 'mc0_by_count'].mean()
print(f'  SPEC minus bulk ISD+: {spec - bulk:+.1f} points')

table = organs.pivot_table(index='organ', columns='method',
                          values='mc0_by_count', aggfunc='mean').reindex(ORGAN_ORDER)
print('\nOrgans, fully cleaved by count [%]:')
print(table.round(2).to_string())
print('SPEC minus PAC: ' + ', '.join(
    f'{o} {table.loc[o, "SPEC"] - table.loc[o, "PAC"]:+.1f}' for o in ORGAN_ORDER))


# ---------------------------------------------------------------------------
# Plot — two sub-panels, mirroring a | b above.
# ---------------------------------------------------------------------------
def draw_points(ax, xpos, values, color):
    """Mean as a horizontal bar with the individual runs as points, as in a and b."""
    ax.scatter(np.full(len(values), xpos)
               + rng.uniform(-0.075, 0.075, size=len(values)), values,
               s=POINT_SIZE, color=color, alpha=0.85, edgecolor='black',
               linewidth=0.3, zorder=4)
    ax.plot(xpos, np.mean(values), marker='_', markersize=10,
            markeredgecolor=color, markeredgewidth=2.2, zorder=5)


rng = np.random.default_rng(0)
# Same width as panel a and the same height as panel c, so the figure's four
# panels sit level; the mean bar is a 10 pt marker, matching a and b exactly.
fig, ax = plt.subplots(figsize=(2.2, 3.2))
for i, condition in enumerate(ORDER):
    draw_points(ax, i, ffpe.loc[ffpe['condition'] == condition,
                                'mc0_by_count'].to_numpy(), COLOR[condition])
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([XTICK[c] for c in ORDER], fontsize=FONTSIZE, rotation=45,
                   ha='right')
ax.set_xlim(-0.6, len(ORDER) - 0.4)
ax.set_ylabel(YLABEL, fontsize=FONTSIZE)
ax.set_ylim(*YLIM)
ax.set_yticks(YTICKS)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')

cols = ['series', 'experiment', 'group', 'run', 'precursors', 'mc0_by_count',
        'mc0_by_intensity']
out = pd.concat([
    ffpe.assign(series='run', experiment='H032_E127 FFPE preparations',
                group=ffpe['condition'])[cols],
    organs.assign(series='run', experiment='H032_E170 organs',
                  group=organs['organ'] + ' ' + organs['method'])[cols]],
    ignore_index=True)
out.to_csv(os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)

print(f'\nSaved {STEM} to {OUTDIR}')
