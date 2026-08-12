"""Step 13 — Where the regulated proteins sit in the abundance distribution."""
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spec_analytics as core
import common as C

C.init()

FONTSIZE = 10
BIN_WIDTH = 1.0
# Pale fill under a saturated outline: the house pattern for overlapping
# distributions, and what raises the contrast without altering a hue.
FILL_ALPHA = 0.18
LINE_WIDTH = 1.8
TARGET_W_IN, TARGET_H_IN = 3.97, 3.87
AXES_W_IN = 3.06        # starting guess; the fit loop below corrects it
AXES_H_IN = 3.25

df = C.load_filtered()
sample_info = C.load_typed()
sel = C.MarkerSelection(df, sample_info)
st = sel.stats

# Abundance is taken from every quantified protein group, not the coverage-filtered
# matrix, so the axis describes the proteome as measured.
abundance = sel.pg_wide_all.median(axis=1)

groups = {
    'Not significant': st.index[~st['significant']],
    'Up in type I (slow)': st.index[st['significant'] & (st['log2fc'] > 0)],
    'Up in type IIb (fast)': st.index[st['significant'] & (st['log2fc'] < 0)],
}
COLOR = {'Not significant': C.GREY,
         'Up in type I (slow)': C.SLOW_COLOR,
         'Up in type IIb (fast)': C.FAST_COLOR}

values = {k: abundance.reindex(v).dropna() for k, v in groups.items()}
print(f'quantified protein groups            {len(abundance):,}')
print(f'testable (>= {C.MIN_VALID_PER_ARM} values in both arms) {len(st):,} '
      f'({100 * len(st) / len(abundance):.0f}% of them)')
for k, v in values.items():
    print(f'  {k:<24} n = {len(v):>5,}   median log2 {v.median():.2f}')

lo = min(v.min() for v in values.values())
hi = max(v.max() for v in values.values())
bins = np.arange(np.floor(lo / BIN_WIDTH) * BIN_WIDTH,
                 np.ceil(hi / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH, BIN_WIDTH)
print(f'\nbins: width {BIN_WIDTH:g} log2, {len(bins) - 1} bins '
      f'from {bins[0]:g} to {bins[-1]:g}')

fig, ax = plt.subplots(figsize=(4.2, 2.8))
for label in sorted(values, key=lambda k: -len(values[k])):
    v = values[label]
    ax.hist(v, bins=bins, histtype='stepfilled', color=COLOR[label],
            alpha=FILL_ALPHA, edgecolor='none', zorder=2)
    ax.hist(v, bins=bins, histtype='step', color=COLOR[label],
            linewidth=LINE_WIDTH, zorder=3)
# Group medians, as in figure 6h. Drawn over the fills so a median that falls inside
# another group's curve stays readable.
for label, v in values.items():
    ax.axvline(v.median(), color=COLOR[label], linestyle='--', linewidth=1.1,
               zorder=4)

ax.set_xlabel('Median log₂ protein intensity', fontsize=FONTSIZE)
ax.set_ylabel(f'Protein groups\n(bin width {BIN_WIDTH:g} log₂)', fontsize=FONTSIZE)
ax.set_xlim(bins[0], bins[-1])
# 1.30x headroom: the three-entry legend sits top-right and the curves peak
# there, so without it the key prints over the bars.
ax.set_ylim(0, max(np.histogram(v, bins=bins)[0].max()
                   for v in values.values()) * 1.30)
ax.tick_params(labelsize=FONTSIZE)
C.despine(ax)
from matplotlib.patches import Patch                            # noqa: E402
ax.legend(handles=[Patch(facecolor=(*mcolors.to_rgb(COLOR[k]), FILL_ALPHA),
                         edgecolor=COLOR[k], linewidth=LINE_WIDTH,
                         label=f'{k} (n = {len(values[k]):,})')
                   for k in values],
          loc='upper right', frameon=False, fontsize=8, handlelength=1.4,
          handleheight=0.9, handletextpad=0.4, borderpad=0.2, labelspacing=0.3)

fig.tight_layout()
axes_w, axes_h = AXES_W_IN, AXES_H_IN
probe = C.PLOTS / '_fit_probe.pdf'
for _ in range(6):
    core.set_axes_size_inches(fig, ax, w_in=axes_w, h_in=axes_h)
    fig.savefig(probe, bbox_inches='tight')
    with open(probe, 'rb') as handle:
        box = re.search(rb'/MediaBox\s*\[([^\]]*)\]', handle.read())
    x0, y0, x1, y1 = (float(v) for v in box.group(1).split())
    got_w, got_h = (x1 - x0) / 72, (y1 - y0) / 72
    if abs(got_w - TARGET_W_IN) <= 0.01 and abs(got_h - TARGET_H_IN) <= 0.01:
        break
    axes_w += TARGET_W_IN - got_w
    axes_h += TARGET_H_IN - got_h
probe.unlink(missing_ok=True)
print(f'\nsaved {got_w:.2f} x {got_h:.2f} in (target {TARGET_W_IN} x {TARGET_H_IN}), '
      f'data area {axes_w:.2f} x {axes_h:.2f} in')

# How far into the low-abundance end the regulated proteins reach — the number the
# panel exists to support, quoted from the raw values rather than from the bins.
sig = pd.concat([values['Up in type I (slow)'], values['Up in type IIb (fast)']])
med_all = abundance.median()
print(f'\nregulated proteins below the median abundance of the whole proteome: '
      f'{int((sig < med_all).sum()):,} of {len(sig):,} '
      f'({100 * (sig < med_all).mean():.0f}%)')
print(f'lowest-abundance regulated protein: log2 {sig.min():.2f}, '
      f'against {abundance.min():.2f} for the least abundant quantified protein')

C.save_panel(fig, 'panel_g_regulated_abundance', {
    'protein': pd.DataFrame({
        'protein_group': st.index,
        'gene': st['gene'].values,
        'median_log2_intensity': abundance.reindex(st.index).values,
        'log2fc_I_vs_IIb': st['log2fc'].values,
        'padj': st['padj'].values,
        'group': np.where(~st['significant'], 'Not significant',
                          np.where(st['log2fc'] > 0, 'Up in type I (slow)',
                                   'Up in type IIb (fast)')),
    }),
    'group median': pd.DataFrame({
        'group': list(values), 'n': [len(v) for v in values.values()],
        'median_log2_intensity': [v.median() for v in values.values()]}),
})
