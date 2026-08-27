"""Supplementary figure 4a — relative peptide signal against sample volume."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import spec_analytics as core
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..',
                                                'supplementary_figure3',
                                                'scripts')))
import common_suppl3 as cs
import common_figure2 as cf
core.init_plotting()
OUTDIR = _cfg.output_dir_of('supplementary_figure4')
ROOT = os.path.join(cs.FIG2_INPUT, 'H032_E306')
STEM = 'suppl4_a_recovery_volume'
CACHE = cs.cache_path('e306_recovery_shared.parquet',
                      quantity_column=cf.RAW_QUANTITY_COLUMN)
METHODS = ['SPEC', 'PAC', 'ISD+']
DISPLAY = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}
COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}
VOLUMES = {'5uL': 5, '10uL': 10, '40uL': 40, '100uL': 100, '200uL': 200}
LEVELS = list(VOLUMES.values())
EXCLUDE_REPLICATE = 3
BAR_IN = 0.106
UNIT_IN = 0.118
AXES_H_IN = cf.AXES_H_IN
FONTSIZE = cf.FONTSIZE
LEGEND_FONTSIZE = cf.LEGEND_FONTSIZE
def build():
    if os.path.exists(CACHE):
        print(f'reusing cache {os.path.basename(CACHE)}')
        return pd.read_parquet(CACHE)
    per = {}
    for method in METHODS:
        for folder, level in VOLUMES.items():
            path = os.path.join(ROOT, method, folder, f'{folder}.parquet')
            pep = cs.read_peptides(path,
                                   quantity_column=cf.RAW_QUANTITY_COLUMN)
            runs = sorted(pep['Run'].unique())
            if len(runs) != 4:
                raise ValueError(f'{method} {folder}: {len(runs)} runs')
            keep = [r for r in runs if r != runs[EXCLUDE_REPLICATE - 1]]
            per[(method, level)] = pep[pep['Run'].isin(keep)]
        print(f'  {method} done')
    complete = []
    for key, pep in per.items():
        n = pep['Run'].nunique()
        seen = pep.groupby('peptide')['Run'].nunique()
        complete.append(set(seen[seen == n].index))
    shared = set.intersection(*complete)
    print(f'peptides quantified in every run of all {len(per)} conditions: '
          f'{len(shared):,}')
    rows = []
    for (method, level), pep in per.items():
        sub = pep[pep['peptide'].isin(shared)]
        for run, total in sub.groupby('Run')['quantity'].sum().items():
            rows.append(dict(method=method, volume_uL=level, run=run,
                             summed_quantity=float(total),
                             n_shared_peptides=len(shared)))
    out = pd.DataFrame(rows)
    out.to_parquet(CACHE, index=False)
    return out
table = build()
own_ref = (table[table.volume_uL == LEVELS[0]]
           .groupby('method').summed_quantity.mean())
table['pct_of_own_min_volume'] = 100.0 * (table.summed_quantity
                                          / table.method.map(own_ref))
table['pct_of_spec_at_min_volume'] = (100.0 * table.summed_quantity
                                      / float(own_ref['SPEC']))
print('PLOTTED - relative peptide signal [%] of each method at its own 5 uL:')
print(table.pivot_table(index='volume_uL', columns='method',
                        values='pct_of_own_min_volume').round(1).to_string())
print('context, NOT plotted - [%] of SAX SPEC at 5 uL, which carries the '
      'cross-method recovery difference as well:')
print(table.pivot_table(index='volume_uL', columns='method',
                        values='pct_of_spec_at_min_volume').round(1).to_string())
print('\nfold change over the volume range:')
for m in METHODS:
    s = table[table.method == m].groupby('volume_uL').summed_quantity.median()
    print(f'  {DISPLAY[m]:9s} {s.iloc[-1] / s.iloc[0]:.2f}x')
fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
rng = np.random.default_rng(0)
n = len(METHODS)
step = n + cf.GROUP_GAP
jitter = 0.22 * cf.BAR_FRAC
heights, points = [], []
for k, method in enumerate(METHODS):
    for i, level in enumerate(LEVELS):
        vals = table[(table.method == method)
                     & (table.volume_uL == level)].pct_of_own_min_volume
        vals = vals.to_numpy(dtype=float)
        centre = i * step + k + 0.5
        mean = float(np.mean(vals))
        ax.bar(centre, mean, cf.BAR_FRAC, color=COLOR[method],
               edgecolor='black', linewidth=0.7, zorder=2)
        points.append(ax.scatter(
            centre + rng.uniform(-jitter, jitter, size=vals.size), vals,
            s=cf.POINT_SIZE_PLACEHOLDER, color='black', alpha=0.85,
            linewidth=0.3, edgecolor='white', zorder=5))
        heights.append((DISPLAY[method], level, mean))
ax.axhline(100, color='#999999', linestyle=':', linewidth=0.8, zorder=0)
centres = [i * step + n / 2 for i in range(len(LEVELS))]
first_edge = 0.5 - cf.BAR_FRAC / 2
last_edge = (len(LEVELS) - 1) * step + n - 0.5 + cf.BAR_FRAC / 2
ax.set_xlim(first_edge - cf.EDGE_PAD, last_edge + cf.EDGE_PAD)
ax.set_xticks(centres)
ax.set_xticklabels([str(v) for v in LEVELS], fontsize=FONTSIZE)
ax.set_xlabel('Sample volume [µL]', fontsize=FONTSIZE)
ax.set_ylabel('Peptide signal, % of own 5 µL', fontsize=FONTSIZE)
ax.set_ylim(0, 145)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.label.set_fontweight('normal')
ax.yaxis.label.set_fontweight('normal')
ax.tick_params(labelsize=FONTSIZE)
ax.legend(handles=[Patch(facecolor=COLOR[m], edgecolor='black',
                         linewidth=0.7, label=DISPLAY[m]) for m in METHODS],
          loc='upper right', frameon=False, fontsize=LEGEND_FONTSIZE, ncol=3,
          handlelength=1.05, handleheight=0.95, handletextpad=0.4,
          columnspacing=0.9, borderaxespad=0.0)
fig.tight_layout()
core.set_axes_size_inches(fig, ax, h_in=AXES_H_IN)
core.scale_replicate_points(ax, points)
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')
src = table[['method', 'volume_uL', 'run', 'summed_quantity',
             'pct_of_own_min_volume', 'pct_of_spec_at_min_volume',
             'n_shared_peptides']].copy()
src.insert(0, 'series', 'replicate point')
src['method'] = src['method'].map(DISPLAY)
bars = pd.DataFrame([{'series': 'bar height (mean)', 'method': m,
                      'volume_uL': v, 'run': None, 'summed_quantity': None,
                      'pct_of_own_min_volume': h,
                      'pct_of_spec_at_min_volume': '',
                      'n_shared_peptides': int(table.n_shared_peptides.iloc[0])}
                     for m, v, h in heights])
pd.concat([src, bars], ignore_index=True).to_csv(
    os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)
print(f'Saved {STEM} to {OUTDIR}')
