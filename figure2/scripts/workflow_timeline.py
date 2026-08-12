"""Retired — total workflow duration on a common, proportional time axis."""

import os
import sys
import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Patch

import spec_analytics as core
import common_figure2 as cf

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
OUTDIR = _cfg.output_dir(__file__)
STEM = 'workflow_timeline'

# (label, hue, variant note, total hours low, total hours high, provenance).
# lo == hi draws a single flat arrow with no extension.
PROTOCOLS = [
    {'label': 'SAX SPEC', 'color': core.PALETTE_SINGLE[0], 'note': 'this work',
     'lo': 2.0, 'hi': 2.0, 'source': 'this work, routine protocol'},
    {'label': 'PAC', 'color': core.PALETTE_SINGLE[2],
     'note': 'automated / accelerated',
     'lo': 4.0, 'hi': 6.0, 'source': 'reported total, accelerated protocol'},
    {'label': 'ISD', 'color': core.PALETTE_SINGLE[1], 'note': 'no cleanup',
     'lo': 1.0, 'hi': 18.0, 'source': 'reported total, fast to overnight digest'},
    {'label': 'ISD+', 'color': core.PALETTE_SINGLE[5],
     'note': '+ SDB-RPS cleanup',
     'lo': 2.0, 'hi': 19.0, 'source': 'reported total, fast to overnight digest'},
]

REFERENCE_H = 2.0          # SAX SPEC's total, drawn as a guide across every row
RANGE_ALPHA = 0.16         # the optional extension, deliberately the faintest mark
BAR_H = 0.56               # bar height in row units
HEAD_H = 0.76              # arrowhead height in row units
HEAD_W_H = 0.36            # arrowhead length, in HOURS (data units)
AXES_W_IN = 6.55           # data area; row labels and totals live outside it
AXES_H_IN = 1.42           # four rows
PANEL_W_IN = 8.42          # matches the retired artwork and the two-up rows below
PAD_IN = 0.1
XMAX_H = 20.0
XTICKS = [0, 4, 8, 12, 16, 20]


def fmt_hours(value):
    """`1`, `2`, `19` — no trailing zeros, a decimal only where it matters."""
    return f'{value:.1f}'.rstrip('0').rstrip('.')


def total_label(lo, hi):
    return f'{fmt_hours(lo)} h' if abs(hi - lo) < 0.05 \
        else f'{fmt_hours(lo)}–{fmt_hours(hi)} h'


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n = len(PROTOCOLS)
fig, ax = plt.subplots(figsize=(PANEL_W_IN, 2.4))

for i, p in enumerate(PROTOCOLS):
    y = n - 1 - i                      # row 0 at the top
    lo, hi = p['lo'], p['hi']
    ranged = hi - lo > 0.05

    if ranged:
        ax.add_patch(Rectangle((lo, y - BAR_H / 2), hi - lo - HEAD_W_H, BAR_H,
                               facecolor=p['color'], alpha=RANGE_ALPHA,
                               edgecolor='none', zorder=2))
        ax.add_patch(Rectangle((lo, y - BAR_H / 2), hi - lo - HEAD_W_H, BAR_H,
                               facecolor='none', edgecolor=p['color'],
                               linewidth=0.7, linestyle=(0, (2.5, 1.8)),
                               zorder=3))
    # The committed time. A flat protocol keeps its arrowhead's length out of the
    # bar so the tip lands exactly on its total rather than overshooting it.
    body = lo if ranged else max(lo - HEAD_W_H, 0.0)
    ax.add_patch(Rectangle((0, y - BAR_H / 2), body, BAR_H,
                           facecolor=p['color'], alpha=0.85,
                           edgecolor=p['color'], linewidth=0.9, zorder=4))

    head = Polygon([[hi, y], [hi - HEAD_W_H, y - HEAD_H / 2],
                    [hi - HEAD_W_H, y + HEAD_H / 2]], closed=True,
                   facecolor=p['color'], edgecolor=p['color'], linewidth=0.9,
                   zorder=5)
    if ranged:
        head.set_alpha(0.45)           # the tip belongs to the band it terminates
    ax.add_patch(head)

    ax.text(hi + 0.35, y, total_label(lo, hi), va='center', ha='left',
            fontsize=cf.FONTSIZE, color='black')

ax.axvline(REFERENCE_H, color='#4D4D4D', linestyle='--', linewidth=0.8,
           alpha=0.75, zorder=6)
# The guide carries no label of its own: it starts at SAX SPEC's arrowhead, which is
# already labelled "2 h", and a second "2 h" in a panel this small read as clutter.

ax.set_xlim(0, XMAX_H + 4.2)           # room for the total labels outside the axis
ax.set_ylim(-0.72, n - 0.32)
ax.set_xticks(XTICKS)
ax.set_xticklabels([str(t) for t in XTICKS], fontsize=cf.FONTSIZE)
ax.set_yticks(range(n))
ax.set_yticklabels([p['label'] for p in reversed(PROTOCOLS)],
                   fontsize=cf.FONTSIZE)
ax.set_xlabel('Time [h]', fontsize=cf.FONTSIZE)
ax.tick_params(labelsize=cf.FONTSIZE, length=3)
ax.tick_params(axis='y', length=0)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)
# The x spine must stop at the axis proper: it is a time axis, and running it out
# under the total labels implies those labels sit at a time.
ax.spines['bottom'].set_bounds(0, XMAX_H)

# Two-entry key in neutral grey — hue already means the method, so a hue-coloured
# key here would claim to mean both. It only has to explain solid versus pale.
ax.legend(handles=[
    Patch(facecolor=cf.LEGEND_DARK, alpha=0.85, edgecolor=cf.LEGEND_DARK,
          linewidth=0.9, label='shortest reported protocol'),
    Patch(facecolor=cf.LEGEND_DARK, alpha=RANGE_ALPHA, edgecolor=cf.LEGEND_DARK,
          linewidth=0.7, linestyle=(0, (2.5, 1.8)),
          label='up to the longest (overnight digestion)')],
    loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
    fontsize=cf.LEGEND_FONTSIZE, handlelength=1.1, handleheight=0.9,
    handletextpad=0.4, borderpad=0.0, columnspacing=1.2)

fig.tight_layout()
cf.set_axes_size_inches(fig, ax, w_in=AXES_W_IN, h_in=AXES_H_IN)


def pdf_width_inches(path):
    with open(path, 'rb') as handle:
        box = re.search(rb'/MediaBox\s*\[([^\]]*)\]', handle.read())
    x0, _y0, x1, _y1 = (float(v) for v in box.group(1).split())
    return (x1 - x0) / 72.0


# Close the loop on the WRITTEN MediaBox: savefig recomputes the tight box with its
# own renderer, ~0.03 in off whatever `get_tightbbox` reports beforehand.
pdf_path = os.path.join(OUTDIR, f'{STEM}.pdf')
for _ in range(6):
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=PAD_IN)
    width = pdf_width_inches(pdf_path)
    if abs(width - PANEL_W_IN) <= 0.01:
        break
    axes_w = ax.get_position().width * fig.get_size_inches()[0]
    cf.set_axes_size_inches(fig, ax, w_in=axes_w + (PANEL_W_IN - width),
                            h_in=AXES_H_IN)
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight',
            pad_inches=PAD_IN)

print(f'{STEM}: saved at {width:.2f} in wide (target {PANEL_W_IN}), '
      f'{fig.get_size_inches()[1]:.2f} in tall')
spec = PROTOCOLS[0]
print(f'\ntotals, and how they compare with SAX SPEC ({fmt_hours(spec["lo"])} h):')
for p in PROTOCOLS:
    label = total_label(p['lo'], p['hi'])
    ratio = ('—' if p is spec
             else f'{p["lo"] / spec["lo"]:.1f}x to {p["hi"] / spec["hi"]:.1f}x')
    print(f'  {p["label"]:9s} {p["note"]:24s} {label:>8s}   {ratio}')

pd.DataFrame([{'protocol': p['label'], 'variant': p['note'],
               'total_shortest_h': p['lo'], 'total_longest_h': p['hi'],
               'source': p['source']} for p in PROTOCOLS]).to_csv(
    os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)
print(f'\nSaved {STEM} to {OUTDIR}')
