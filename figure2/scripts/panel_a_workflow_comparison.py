"""RETIRED — qualitative workflow comparison of SPEC, PAC, ISD+ and ISD."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
OUTDIR = _cfg.output_dir(__file__)

STEPS = ['Sample lysis', 'Sample preparation', 'Digestion', 'Cleanup',
         'Sample recovery', 'LC-MS analysis', 'Scalability']

# Levels per method, in the STEPS order shown above the numbers.
SCORES = {
    #        lysis  prep  dig  clean  recov  LC-MS  scale
    'SPEC':  [  4,    3,    3,    4,     4,     4,     4],
    'PAC':   [  3,    2,    3,    4,     2,     4,     4],
    'ISD+':  [  4,    4,    4,    2,     1,     3,     1],
    'ISD':   [  1,    4,    4,    1,     3,     1,     2],
}

# Headers wrap onto two lines; on one line the neighbouring ones collide.
HEADER = {'Sample lysis': 'Sample\nlysis',
          'Sample preparation': 'Sample\npreparation',
          'Digestion': 'Digestion', 'Cleanup': 'Cleanup',
          'Sample recovery': 'Sample\nrecovery',
          'LC-MS analysis': 'LC-MS\nanalysis',
          'Scalability': 'Scalability',
          'Overall': 'Overall'}

# Four ordinal levels, bad -> good (ColorBrewer RdBu-4).
LEVELS = [1, 2, 3, 4]
LEVEL_NAME = {1: 'Limiting', 2: 'Acceptable', 3: 'Good', 4: 'Optimal'}
LEVEL_COLORS = ['#B2182B', '#EF8A62', '#67A9CF', '#2166AC']
SHOW_OVERALL = True
GAP = 0.6                 # blank width before the Overall cell, in cell units
LABEL_FONTSIZE = 8

# Rows run best to worst overall, so the order follows an edited score.
METHODS = sorted(SCORES, key=lambda m: np.mean(SCORES[m]), reverse=True)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
levels = np.array([SCORES[m] for m in METHODS], float)
overall = levels.mean(axis=1)
# Rounded to the same four levels (down on an exact .5): a continuous ramp between
# levels passes through grey near 2.5, adding a colour the legend cannot explain.
overall_level = np.floor(overall + 0.5 - 1e-9)

n_rows = len(METHODS)
fig, ax = plt.subplots(figsize=(8.5, 1.9))

blocks = [(0.0, STEPS, levels)]
if SHOW_OVERALL:
    blocks.append((len(STEPS) + GAP, ['Overall'], overall_level[:, None]))

xticks, xlabels = [], []
for x0, names, values in blocks:
    for i in range(n_rows):
        for j in range(len(names)):
            ax.add_patch(Rectangle(
                (x0 - 0.5 + j, i - 0.5), 1, 1,
                facecolor=LEVEL_COLORS[int(values[i, j]) - 1],
                edgecolor='white', linewidth=2, zorder=2))
    xticks += [x0 + j for j in range(len(names))]
    xlabels += [HEADER[n] for n in names]

ax.set_xlim(-0.5, blocks[-1][0] - 0.5 + len(blocks[-1][1]))
ax.set_ylim(n_rows - 0.5, -0.5)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=LABEL_FONTSIZE)
ax.xaxis.set_ticks_position('top')
ax.set_yticks(range(n_rows))
ax.set_yticklabels(METHODS, fontsize=LABEL_FONTSIZE)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.legend(handles=[Patch(facecolor=LEVEL_COLORS[k - 1], label=LEVEL_NAME[k])
                   for k in LEVELS],
          loc='upper left', bbox_to_anchor=(0.0, -0.06), ncol=len(LEVELS),
          frameon=False, fontsize=LABEL_FONTSIZE, handlelength=0.9,
          handleheight=0.9, handletextpad=0.4, columnspacing=1.1, borderpad=0.0)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'panel_a_workflow_comparison.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_a_workflow_comparison.png'), dpi=300,
            bbox_inches='tight')

# ---------------------------------------------------------------------------
# Source data — the levels as drawn.
# ---------------------------------------------------------------------------
rows = [{'method': m, 'metric': s, 'level': int(SCORES[m][j]),
         'level_name': LEVEL_NAME[SCORES[m][j]]}
        for m in METHODS for j, s in enumerate(STEPS)]
rows += [{'method': m, 'metric': 'Overall', 'level': int(lv),
          'level_name': LEVEL_NAME[int(lv)]}
         for m, lv in zip(METHODS, overall_level)]
pd.DataFrame(rows).to_csv(
    os.path.join(OUTDIR, 'panel_a_workflow_comparison_sourcedata.csv'), index=False)

table = pd.DataFrame(levels, index=METHODS, columns=STEPS)
table['mean'] = overall.round(2)
table['Overall shown'] = [LEVEL_NAME[int(v)] for v in overall_level]
print(table.to_string())
print(f'\nSaved panel a to {OUTDIR}')
