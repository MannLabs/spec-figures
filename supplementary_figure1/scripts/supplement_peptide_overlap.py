"""Supplementary figure 1 — cross-method peptide recovery (H032_E297)."""

import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
BASE = os.path.join(INPUT, 'H032_E297')
OUTDIR = _cfg.output_dir(__file__)

# (folder, parquet stem, label); the ISD+ folder holds ISD_SDB-RPS.* outputs.
CONDITIONS = [
    ('C18',  'C18',         'C18'),
    ('ISD+', 'ISD_SDB-RPS', 'ISD+'),
    ('SCX',  'SCX',         'SCX'),
    ('ISD',  'ISD',         'ISD'),
    ('SAX',  'SAX',         'SAX'),
]
QVALUE = 0.01
N_REPLICATES = 4
SAX_COLOR = core.PALETTE_SINGLE[0]   # coral

# ---------------------------------------------------------------------------
# Peptidoform sets per condition.
# ---------------------------------------------------------------------------
sets = {}
for folder, stem, label in CONDITIONS:
    path = os.path.join(BASE, folder, f'{stem}.parquet')
    raw = pd.read_parquet(path, columns=['Run', 'Modified.Sequence', 'PG.Q.Value'])
    n_runs = raw['Run'].nunique()
    if n_runs != N_REPLICATES:
        raise ValueError(f'{label}: expected {N_REPLICATES} runs in {path}, found {n_runs}')
    sets[label] = set(raw.loc[raw['PG.Q.Value'] < QVALUE, 'Modified.Sequence'].dropna())
    print(f'{label:5s} {n_runs} runs, {len(sets[label]):>8,} peptidoforms')

labels = [lab for _f, _s, lab in CONDITIONS]

# Containment matrix, ordered by the column margin (mean over other methods).
mat = pd.DataFrame(
    {col: {row: 100 * len(sets[row] & sets[col]) / len(sets[row]) for row in labels}
     for col in labels}
)
margin = ((mat.sum(axis=0) - 100) / (len(labels) - 1))
order = list(margin.sort_values().index)
mat = mat.loc[order, order]
margin = margin[order]

print('\n% of ROW method peptides also detected by COLUMN method')
print(mat.round(1).to_string())
print('\nmean over the other methods (column margin):')
print(margin.round(1).to_string())

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
vals = mat.to_numpy(dtype=float)

cmap = LinearSegmentedColormap.from_list(
    'sax_seq', ['#FFFFFF', SAX_COLOR, '#7A2A1C'])
norm = Normalize(vmin=40, vmax=90)
DIAGONAL_FILL = '#F0F0F0'

fig, ax = plt.subplots(figsize=(4, 4))
# Cells as vector rectangles, not imshow: imshow embeds the grid as a bitmap,
# which is what rendered the colours wrong in figure 2a.
for i in range(len(order)):
    for j in range(len(order)):
        face = DIAGONAL_FILL if i == j else cmap(norm(vals[i, j]))
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                               edgecolor='none', zorder=1))
ax.set_xlim(-0.5, len(order) - 0.5)
ax.set_ylim(len(order) - 0.5, -0.5)
ax.set_aspect('equal')

for i, row in enumerate(order):
    for j, col in enumerate(order):
        if i == j:
            ax.text(j, i, f'{len(sets[row]):,}', ha='center', va='center',
                    fontsize=6.5, color='#666666')
        else:
            v = vals[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=9,
                    color='white' if v > 72 else 'black')

# Cell separators.
for k in range(len(order) + 1):
    ax.axhline(k - 0.5, color='white', linewidth=1.5)
    ax.axvline(k - 0.5, color='white', linewidth=1.5)

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=10)
ax.set_xlabel('also detected by', fontsize=10)
ax.set_ylabel('peptides identified by', fontsize=10)
# Tick labels are padded down to leave room for the column-margin row.
ax.tick_params(length=0)
ax.tick_params(axis='x', pad=16)
for s in ax.spines.values():
    s.set_visible(False)

# Column-margin row, maximum bolded.
best = margin.idxmax()
ax.text(-0.60, len(order) - 0.34, 'mean of others', fontsize=7.5,
        ha='right', va='center', color='#333333')
for j, col in enumerate(order):
    ax.text(j, len(order) - 0.34, f'{margin[col]:.0f}', ha='center', va='center',
            fontsize=9, fontweight='bold' if col == best else 'normal',
            color=SAX_COLOR if col == best else '#333333')

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
cb.set_label('% of the row method\'s\npeptides also detected', fontsize=7.5)
cb.ax.tick_params(labelsize=7.5)
cb.outline.set_visible(False)
# Colorbars are drawn as images by default; force vector like figure 6g.
cb.solids.set_rasterized(False)

PANEL_WIDTH_IN = 3.55
PAD_IN = 0.1        # matplotlib's default bbox_inches='tight' padding, per side


def pdf_width_inches(path):
    with open(path, 'rb') as handle:
        box = re.search(rb'/MediaBox\s*\[([^\]]*)\]', handle.read())
    x0, _y0, x1, _y1 = (float(v) for v in box.group(1).split())
    return (x1 - x0) / 72.0


fig.tight_layout()
pdf_path = os.path.join(OUTDIR, 'supplement_peptide_overlap.pdf')
for _ in range(5):
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=PAD_IN)
    width = pdf_width_inches(pdf_path)
    if abs(width - PANEL_WIDTH_IN) <= 0.01:
        break
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * PANEL_WIDTH_IN / width, h * PANEL_WIDTH_IN / width)
print(f'heatmap saved at {width:.2f} in wide (target {PANEL_WIDTH_IN})')
fig.savefig(os.path.join(OUTDIR, 'supplement_peptide_overlap.png'), dpi=300,
            bbox_inches='tight', pad_inches=PAD_IN)

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
rows = []
for i, row in enumerate(order):
    for j, col in enumerate(order):
        rows.append({
            'series': 'diagonal_total' if i == j else 'cell',
            'identified_by': row, 'also_detected_by': col,
            'n_peptides_row': len(sets[row]),
            'n_peptides_shared': len(sets[row] & sets[col]),
            'pct_of_row_recovered': round(vals[i, j], 2),
        })
for col in order:
    rows.append({
        'series': 'column_margin', 'identified_by': 'mean of others',
        'also_detected_by': col, 'n_peptides_row': np.nan,
        'n_peptides_shared': np.nan,
        'pct_of_row_recovered': round(margin[col], 2),
    })
pd.DataFrame(rows).to_csv(
    os.path.join(OUTDIR, 'supplement_peptide_overlap_sourcedata.csv'), index=False)

print(f'\nSaved supplement to {OUTDIR}')
