"""Figure 3c — protein-group overlap between mTRAQ channels (H032_E229)."""
import os
import sys
import re
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
import spec_analytics as core
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
RAW = INPUT
OUTDIR = _cfg.output_dir_of('figure3')
os.makedirs(OUTDIR, exist_ok=True)
ROW_CHANNEL = {'B': '0', 'C': '4', 'D': '8'}
CHANNELS = ['0', '4', '8']
LABELS = {'0': 'd0', '4': 'd4', '8': 'd8'}
QVALUE = 0.01
CHANNEL_COLOR = {'0': core.PALETTE_SINGLE[0],
                 '4': core.PALETTE_SINGLE[2],
                 '8': core.PALETTE_SINGLE[5]}
OVERLAP_DARKEN = 0.16
PANEL_HEIGHT_IN = 3.20
PAD_IN = 0.1
d = pd.read_parquet(os.path.join(RAW, 'H032_E229.parquet'), columns=[
    'Run', 'Channel', 'Protein.Group', 'Q.Value', 'Channel.Q.Value', 'PG.Q.Value'])
d['own_channel'] = d['Run'].str[-2].map(ROW_CHANNEL)
d = d[(d['Channel'] == d['own_channel'])
      & (d['Q.Value'] < QVALUE)
      & (d['Channel.Q.Value'] < QVALUE)
      & (d['PG.Q.Value'] < QVALUE)]
sets = {ch: set(d.loc[d['own_channel'] == ch, 'Protein.Group'].unique())
        for ch in CHANNELS}
for ch in CHANNELS:
    n_runs = d.loc[d['own_channel'] == ch, 'Run'].nunique()
    print(f'{LABELS[ch]}: {n_runs} runs, {len(sets[ch]):,} protein groups')
shared_all = set.intersection(*sets.values())
union_all = set.union(*sets.values())
print(f'\nshared by all three: {len(shared_all):,} '
      f'({100 * len(shared_all) / len(union_all):.1f}% of the union of {len(union_all):,})')
fig, ax = plt.subplots(figsize=(4, 4))
v = venn3(
    [sets[c] for c in CHANNELS],
    set_labels=[f'{LABELS[c]}\n{len(sets[c]):,}' for c in CHANNELS], ax=ax,
    layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,) * 7))
def region_color(region_id):
    """Blend of the contributing sets' hues, darkened by how many contribute.
    `region_id` is matplotlib_venn's 3-bit membership string, e.g. '101' for the
    d0 & d8 region. Averaging the hues gives each region an intermediate colour,
    and the darkening makes overlap depth readable independently of hue.
    """
    members = [CHANNEL_COLOR[ch] for ch, bit in zip(CHANNELS, region_id)
               if bit == '1']
    rgb = np.mean([mcolors.to_rgb(c) for c in members], axis=0)
    return tuple(rgb * (1 - OVERLAP_DARKEN * (len(members) - 1)))
def readable_text_color(rgb):
    """Black or white, whichever contrasts with `rgb` (Rec. 601 luminance)."""
    r, g, b = rgb
    return 'black' if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else 'white'
region_colors = {}
for region_id in ('100', '010', '001', '110', '101', '011', '111'):
    patch = v.get_patch_by_id(region_id)
    if patch is None:
        continue
    region_colors[region_id] = region_color(region_id)
    patch.set_color(region_colors[region_id])
    patch.set_alpha(1.0)
    patch.set_edgecolor('black')
    patch.set_linewidth(0.5)
for text in v.set_labels:
    if text is not None:
        text.set_fontsize(8.5)
for region_id in ('100', '010', '001', '110', '101', '011', '111'):
    text = v.get_label_by_id(region_id)
    if text is None:
        continue
    text.set_fontsize(8)
    text.set_text(f'{int(text.get_text()):,}')
    text.set_color(readable_text_color(region_colors[region_id]))
def fit_saved_height_inches(fig, saved_target_in, *, pad_in=PAD_IN, passes=6,
                            tol=0.004):
    """Scale the figure so the **saved** PDF is `saved_target_in` inches tall.
    `bbox_inches='tight'` writes the tight bounding box plus `pad_inches` on every
    side, so the target for the box itself is the saved height minus twice the pad
    — miss that and the panel comes out 0.2 in taller than asked for.
    Width scales with the height, so the circles stay circular. Iterative rather
    than one ratio because the set labels are text at a fixed point size: they do
    not shrink with the drawing, so the tight box is not proportional to the figure
    size. Returns the achieved saved height.
    """
    box_target = saved_target_in - 2 * pad_in
    height = None
    for _ in range(passes):
        fig.canvas.draw()
        height = fig.get_tightbbox(fig.canvas.get_renderer()).height
        if abs(height - box_target) <= tol:
            break
        scale = box_target / height
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * scale, h * scale)
    return height + 2 * pad_in
def pdf_size_inches(path):
    """Page size of a one-page PDF, read back from its MediaBox."""
    with open(path, 'rb') as handle:
        box = re.search(rb'/MediaBox\s*\[([^\]]*)\]', handle.read())
    x0, y0, x1, y1 = (float(v) for v in box.group(1).split())
    return (x1 - x0) / 72.0, (y1 - y0) / 72.0
fig.tight_layout()
fit_saved_height_inches(fig, PANEL_HEIGHT_IN)
pdf_path = os.path.join(OUTDIR, 'panel_c_channel_overlap.pdf')
for _ in range(4):
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=PAD_IN)
    width_in, height_in = pdf_size_inches(pdf_path)
    if abs(height_in - PANEL_HEIGHT_IN) <= 0.005:
        break
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * PANEL_HEIGHT_IN / height_in,
                        h * PANEL_HEIGHT_IN / height_in)
print(f'saved {width_in:.3f} x {height_in:.3f} in '
      f'(target height {PANEL_HEIGHT_IN}, matching panel b)')
fig.savefig(os.path.join(OUTDIR, 'panel_c_channel_overlap.png'), dpi=300,
            bbox_inches='tight')
rows = []
for ch in CHANNELS:
    others = set.union(*[sets[c] for c in CHANNELS if c != ch])
    rows.append({'region': f'{LABELS[ch]} only', 'n_protein_groups': len(sets[ch] - others)})
for a, b in combinations(CHANNELS, 2):
    third = [c for c in CHANNELS if c not in (a, b)][0]
    rows.append({'region': f'{LABELS[a]} & {LABELS[b]} only',
                 'n_protein_groups': len((sets[a] & sets[b]) - sets[third])})
rows.append({'region': 'shared by all three', 'n_protein_groups': len(shared_all)})
for ch in CHANNELS:
    rows.append({'region': f'{LABELS[ch]} total', 'n_protein_groups': len(sets[ch])})
rows.append({'region': 'union', 'n_protein_groups': len(union_all)})
pd.DataFrame(rows).to_csv(
    os.path.join(OUTDIR, 'panel_c_channel_overlap_sourcedata.csv'), index=False)
print(f'\nSaved panel c to {OUTDIR}')
