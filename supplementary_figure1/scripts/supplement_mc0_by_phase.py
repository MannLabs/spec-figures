"""Supplementary figure 1 — digestion completeness of the five phases (H032_E297)."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
BASE = os.path.join(INPUT, 'H032_E297')
OUTDIR = _cfg.output_dir(__file__)

# (label, folder, parquet stem, hue) — the ISD+ folder holds ISD_SDB-RPS.* files.
# Order and hues match figure 1c and the sibling panels of this figure.
CONDITIONS = [
    ('C18',  'C18',  'C18',         core.PALETTE_SINGLE[3]),
    ('ISD+', 'ISD+', 'ISD_SDB-RPS', core.PALETTE_SINGLE[5]),
    ('SCX',  'SCX',  'SCX',         core.PALETTE_SINGLE[2]),
    ('ISD',  'ISD',  'ISD',         core.PALETTE_SINGLE[1]),
    ('SAX',  'SAX',  'SAX',         core.PALETTE_SINGLE[0]),
]
ORDER = [c[0] for c in CONDITIONS]
COLOR = {c[0]: c[3] for c in CONDITIONS}

QVALUE = 0.01
N_REPLICATES = 4
FONTSIZE = 8
STEM = 'supplement_mc0_by_phase'

BAR_IN = 0.29
UNIT_IN = 0.385
# 2.35 rather than the paper-wide 2.50, so the saved height matches the heatmap
# beside it once both are shrunk.
AXES_H_IN = 2.35
POINT_SIZE = core.replicate_point_size(BAR_IN)


def set_bar_geometry_inches(fig, ax, *, bar_in=BAR_IN, unit_in=UNIT_IN,
                            h_in=AXES_H_IN):
    """Pin the axes to `unit_in` per x unit and `h_in` tall, then set bar widths.

    Call after `tight_layout()`; the margins it measured are converted to inches
    and kept, so only the data area is resized and every label keeps its
    clearance. `bbox_inches='tight'` crops without scaling, so the saved PDF keeps
    these inch values.
    """
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    left_in, right_in = pos.x0 * fig_w, (1.0 - pos.x1) * fig_w
    bottom_in, top_in = pos.y0 * fig_h, (1.0 - pos.y1) * fig_h
    x0, x1 = ax.get_xlim()

    axes_w = (x1 - x0) * unit_in
    new_w, new_h = left_in + axes_w + right_in, bottom_in + h_in + top_in
    fig.set_size_inches(new_w, new_h)
    ax.set_position([left_in / new_w, bottom_in / new_h,
                     axes_w / new_w, h_in / new_h])

    w_data = bar_in / unit_in
    for patch in ax.patches:
        centre = patch.get_x() + patch.get_width() / 2
        patch.set_width(w_data)
        patch.set_x(centre - w_data / 2)


# ---------------------------------------------------------------------------
# Per-run fully-cleaved rates.
# ---------------------------------------------------------------------------
mc_memo = {}
rows = []
for label, folder, stem, _color in CONDITIONS:
    d = pd.read_parquet(os.path.join(BASE, folder, f'{stem}.parquet'), columns=[
        'Run', 'Stripped.Sequence', 'Precursor.Quantity', 'Q.Value', 'PG.Q.Value'])
    d = d[(d['Q.Value'] < QVALUE) & (d['PG.Q.Value'] < QVALUE)
          & (d['Precursor.Quantity'] > 0)]
    if d['Run'].nunique() != N_REPLICATES:
        raise ValueError(f'{label}: expected {N_REPLICATES} runs, '
                         f'found {d["Run"].nunique()}')
    for seq in d['Stripped.Sequence'].unique():
        if seq not in mc_memo:
            mc_memo[seq] = core.count_missed_cleavages(seq, protease='trypsin')
    d = d.assign(is_mc0=d['Stripped.Sequence'].map(mc_memo) == 0)

    for replicate, (run, g) in enumerate(sorted(d.groupby('Run')), start=1):
        rows.append({
            'condition': label, 'run': run, 'replicate': replicate,
            'mc0_by_count': 100 * g['is_mc0'].mean(),
            'mc0_by_intensity': (100 * g.loc[g['is_mc0'], 'Precursor.Quantity'].sum()
                                 / g['Precursor.Quantity'].sum()),
            'precursors': int(len(g)),
        })

per_run = pd.DataFrame(rows)
summary = (per_run.groupby('condition')[['mc0_by_count', 'mc0_by_intensity']]
           .agg(['mean', 'std']).reindex(ORDER))
print('fully-cleaved rate [%] (mean, SD over 4 replicates):')
print(summary.round(2).to_string())
gap = (summary[('mc0_by_intensity', 'mean')] - summary[('mc0_by_count', 'mean')])
print('\nintensity minus count [percentage points]:')
print(gap.round(2).to_string())

# ---------------------------------------------------------------------------
# Plot — count-weighted rate, one bar per phase.
# ---------------------------------------------------------------------------
x = np.arange(len(ORDER))
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(3.2, 4))

for i, cond in enumerate(ORDER):
    vals = per_run.loc[per_run['condition'] == cond, 'mc0_by_count'].to_numpy()
    ax.bar(x[i], vals.mean(), 0.6, color=COLOR[cond], edgecolor='black',
           linewidth=0.8, zorder=2)
    ax.scatter(x[i] + rng.uniform(-0.07, 0.07, size=vals.size), vals,
               s=POINT_SIZE, color='black', alpha=0.85, linewidth=0.3,
               edgecolor='white', zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(ORDER, fontsize=FONTSIZE)
ax.set_xlim(-0.6, len(ORDER) - 0.4)
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel('Fully cleaved precursors [%]', fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
set_bar_geometry_inches(fig, ax)
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Source data — plotted values, plus the intensity-weighted rate for the record.
# ---------------------------------------------------------------------------
cols = ['series', 'condition', 'run', 'replicate', 'mc0_by_count',
        'mc0_by_intensity', 'precursors']
bars = pd.DataFrame([
    {'series': 'bar height (mean)', 'condition': c, 'run': '',
     'replicate': np.nan,
     'mc0_by_count': per_run.loc[per_run['condition'] == c,
                                 'mc0_by_count'].mean(),
     'mc0_by_intensity': per_run.loc[per_run['condition'] == c,
                                     'mc0_by_intensity'].mean(),
     'precursors': np.nan}
    for c in ORDER])
pd.concat([per_run.assign(series='replicate point')[cols], bars[cols]],
          ignore_index=True).to_csv(
    os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)

print(f'\nSaved {STEM} to {OUTDIR}')
