"""Supplementary figure 1 — digestion completeness of the five phases (H032_E297)."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import spec_analytics as core
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
BASE = os.path.join(INPUT, 'H032_E297')
OUTDIR = _cfg.output_dir(__file__)
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
AXES_H_IN = 2.35
POINT_SIZE = core.replicate_point_size(BAR_IN)
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
GROUPS = [('in solution', ['ISD+', 'ISD']), ('SPEC', ['C18', 'SCX', 'SAX'])]
means = {c: float(per_run.loc[per_run['condition'] == c,
                              'mc0_by_intensity'].mean()) for c in ORDER}
points = {c: per_run.loc[per_run['condition'] == c,
                         'mc0_by_intensity'].to_numpy() for c in ORDER}
fig, ax = plt.subplots(figsize=(3.2, 4))
core.plot_grouped_bars(
    GROUPS, means, colors={c: COLOR[c] for c in ORDER}, points=points,
    ax=ax, y_label='Fully cleaved signal [%]', ylim=(0, 100),
    bar_in=BAR_IN, unit_in=UNIT_IN, legend=False,
    point_size=POINT_SIZE,
    tick_fontsize=FONTSIZE, label_fontsize=FONTSIZE)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.tick_params(labelsize=FONTSIZE)
BAR_ORDER = [c for _g, members in GROUPS for c in members]
ax.legend(handles=[Patch(facecolor=COLOR[c], edgecolor='black', linewidth=0.7,
                         label=c) for c in BAR_ORDER],
          loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=5,
          frameon=False, fontsize=FONTSIZE, handlelength=0.9,
          handletextpad=0.4, columnspacing=0.9, borderpad=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
core.fix_bar_geometry(fig, ax, bar_in=BAR_IN, unit_in=UNIT_IN, h_in=AXES_H_IN)
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')
cols = ['series', 'group', 'condition', 'run', 'replicate', 'mc0_by_count',
        'mc0_by_intensity', 'precursors']
group_of = {c: g for g, members in GROUPS for c in members}
per_run = per_run.assign(group=per_run['condition'].map(group_of))
bars = pd.DataFrame([
    {'series': 'bar height (mean)', 'group': group_of[c], 'condition': c,
     'run': '', 'replicate': np.nan,
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
