"""Figure 1f — how much of a plain in-solution digest's peptide space each format"""
import glob
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase
import spec_analytics as core
core.init_plotting()
PANEL_LEFT_IN = 0.81
def save_matched(fig, ax, stem):
    """Save PDF + PNG with the axes' left edge PANEL_LEFT_IN from the box edge."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    gutter = ax.get_position().x0 * fig.get_size_inches()[0] - tb.x0
    pad = max(0.0, PANEL_LEFT_IN - gutter)
    box = Bbox.from_extents(tb.x0 - pad, tb.y0, tb.x1, tb.y1)
    fig.savefig(os.path.join(OUTDIR, f'{stem}.pdf'), bbox_inches=box)
    fig.savefig(os.path.join(OUTDIR, f'{stem}.png'), dpi=300, bbox_inches=box)
    print(f'  {stem}: gutter {gutter:.3f} in, padded {pad:.3f}, '
          f'box {box.width:.3f} in')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
OUTDIR = _cfg.output_dir(__file__)
S1_INPUT = _cfg.cross_input('supplementary_figure1')
REPORTS = os.path.join(S1_INPUT, 'H032_E297')
STEM = 'panel_f_peptide_overlap'
QVALUE = 0.01
N_REPLICATES = 4
PEPTIDE_COLUMN = 'Modified.Sequence'
REFERENCE = 'ISD'
GREY = '#c8c8c8'
BAR_FRAC = 0.74
BAR_IN = 0.195
UNIT_IN = 0.26
AXES_H_IN = 2.50
FONTSIZE = 10
LEGEND_FONTSIZE = 10
CONDITIONS = [
    ('ISD+', 'ISD+', core.PALETTE_SINGLE[5]),
    ('ISD',  'ISD',  core.PALETTE_SINGLE[1]),
    ('C18',  'C18',  core.PALETTE_SINGLE[3]),
    ('SCX',  'SCX',  core.PALETTE_SINGLE[2]),
    ('SAX',  'SAX',  core.PALETTE_SINGLE[0]),
]
COLOR = {label: color for label, _folder, color in CONDITIONS}
GROUPS = [('in solution', ['ISD+', 'ISD']), ('SPEC', ['C18', 'SCX', 'SAX'])]
ORDER = [c for _g, members in GROUPS for c in members]
class HandlerStripes(HandlerBase):
    """Legend swatch striped with every format's colour.
    The shared segment is drawn in each bar's own hue, so a single-colour swatch
    would have to pick one format and would read as being about that format.
    """
    def __init__(self, colors):
        super().__init__()
        self.colors = colors
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        n = len(self.colors)
        return [Rectangle((xdescent + i * width / n, ydescent), width / n,
                          height, facecolor=c, edgecolor='none',
                          transform=trans)
                for i, c in enumerate(self.colors)]
def peptide_set(folder):
    """Peptidoforms identified in a condition, unioned over its replicates."""
    hits = sorted(glob.glob(os.path.join(REPORTS, folder, '*.parquet')))
    if len(hits) != 1:
        raise FileNotFoundError(
            f'expected one parquet under {os.path.join(REPORTS, folder)}, '
            f'found {len(hits)}')
    path = hits[0]
    t = pq.read_table(
        path, columns=['Run', PEPTIDE_COLUMN, 'Precursor.Quantity',
                       'Q.Value', 'PG.Q.Value'],
        filters=[('Q.Value', '<=', QVALUE), ('PG.Q.Value', '<=', QVALUE),
                 ('Precursor.Quantity', '>', 0)]).to_pandas()
    if t['Run'].nunique() != N_REPLICATES:
        raise ValueError(f'{folder}: {t["Run"].nunique()} runs, '
                         f'expected {N_REPLICATES}')
    return set(t[PEPTIDE_COLUMN].unique())
sets = {label: peptide_set(folder) for label, folder, _c in CONDITIONS}
ref = sets[REFERENCE]
rows = []
for label in ORDER:
    s = sets[label]
    shared = len(s & ref)
    rows.append(dict(condition=label, peptides=len(s), shared_with_isd=shared,
                     not_shared_with_isd=len(s) - shared,
                     pct_of_isd_recovered=100.0 * shared / len(ref),
                     pct_of_own_shared=100.0 * shared / len(s)))
summary = pd.DataFrame(rows).set_index('condition')
print(summary.round(1).to_string())
YMAX = float(summary['peptides'].max()) * 1.42
fig, ax = plt.subplots(figsize=(3.2, 4))
_f, _a, pos = core.plot_grouped_bars(
    GROUPS, {c: float(summary.loc[c, 'shared_with_isd']) for c in ORDER},
    colors={c: COLOR[c] for c in ORDER}, ax=ax, y_label='Peptides',
    ylim=(0, YMAX), bar_in=BAR_IN, unit_in=UNIT_IN, legend=False,
    tick_fontsize=FONTSIZE, label_fontsize=FONTSIZE)
seg_width = ax.patches[0].get_width()
for c in ORDER:
    extra = float(summary.loc[c, 'not_shared_with_isd'])
    if extra > 0:
        ax.bar(pos[c], extra, seg_width,
               bottom=float(summary.loc[c, 'shared_with_isd']), color=GREY,
               edgecolor='black', linewidth=0.7, zorder=3)
ax.axhline(len(ref), color='#555555', linestyle='--', linewidth=0.9, zorder=1)
ax.set_yticks(np.arange(0, 175001, 50000))
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(
    lambda v, _p: '0' if v == 0 else f'{v/1000:g}k'))
ax.tick_params(labelsize=FONTSIZE)
KW = dict(frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=0.9,
          handletextpad=0.4, columnspacing=0.9, borderpad=0.2)
formats = ax.legend(
    handles=[Patch(facecolor=COLOR[c], edgecolor='black', linewidth=0.7,
                   label=c) for c in ORDER],
    loc='upper center', ncol=3, **KW)
ax.add_artist(formats)
ax.legend(handles=[Patch(facecolor=GREY, edgecolor='black', linewidth=0.7,
                         label='not shared with ISD')],
          loc='upper center', bbox_to_anchor=(0.5, 0.845), ncol=1, **KW)
fig.tight_layout()
core.fix_bar_geometry(fig, ax, bar_in=BAR_IN, unit_in=UNIT_IN, h_in=AXES_H_IN)
save_matched(fig, ax, STEM)
src = summary.reset_index()
src.insert(0, 'series', 'bar segments')
src.to_csv(os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)
print(f'\nreference ({REFERENCE}): {len(ref):,} peptides')
print(f'Saved {STEM} to {OUTDIR}')
