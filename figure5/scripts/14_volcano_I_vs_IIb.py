"""Step 14 — Single-fiber volcano: pure type I vs pure type IIb, with the"""
import matplotlib.pyplot as plt
import pandas as pd

import spec_analytics as core
import common as C

C.init()

df = C.load_filtered(columns=C.DE_COLUMNS)
sample_info = C.load_typed()

runs = C.type_runs(sample_info)
n_I, n_IIb = len(runs['I']), len(runs['IIb'])
de_df = C.de_input(df, sample_info)
print(f'pure I = {n_I}, pure IIb = {n_IIb}')
print(f'protein groups after coverage filter: {de_df["protein_group"].nunique():,}')

fig, ax, vdf = core.plot_volcano(
    de_df, sample_info, **C.DE_KWARGS,
    palette=[C.SLOW_COLOR, C.FAST_COLOR],     # Up = up in I (slow), Down = IIb
    highlight_genes=sorted(C.TEXTBOOK_DE),
    highlight_color=C.TEXTBOOK_COLOR,
    highlight_size=10,               # colour is the highlight, not dot size
    label_highlighted=False,                  # placed in the margins instead
    figsize=(4, 4),
    point_size_sig=10, point_size_ns=5,
    title=f'Single fibers · type I vs type IIb · n={n_I}/{n_IIb}',
)
C.unbold(ax)

for coll in ax.collections:
    if str(coll.get_label()).startswith('Highlighted'):
        offsets = coll.get_offsets()
        coll.set_facecolor([C.SLOW_COLOR if x > 0 else C.FAST_COLOR
                            for x, _y in offsets])
        coll.set_edgecolor(C.TEXTBOOK_COLOR)
        coll.set_linewidths(1.1)
        coll.set_sizes([26] * len(offsets))   # room for the ring to read
        coll.set_zorder(5)

# Rename the legend so it reads biologically rather than as Up/Down.
handles, labels = ax.get_legend_handles_labels()
# Swap the marker key for one that shows the ring rather than a lavender disc.
from matplotlib.lines import Line2D          # noqa: E402
for i, lab in enumerate(labels):
    if lab.split(' (')[0] == 'Highlighted':
        handles[i] = Line2D([0], [0], linestyle='none', marker='o', markersize=5,
                            markerfacecolor=C.GREY,
                            markeredgecolor=C.TEXTBOOK_COLOR, markeredgewidth=1.1)
rename = {'Up': f'Up in type I (slow)', 'Down': 'Up in type IIb (fast)',
          'NS': 'Not significant', 'Highlighted': 'Textbook marker'}
labels = [f"{rename.get(l.split(' (')[0], l.split(' (')[0])} ({l.split('(')[1]}"
          if '(' in l else l for l in labels]
ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.01),
          ncol=2, fontsize=7.5, frameon=False, labelspacing=0.3,
          handletextpad=0.3, borderaxespad=0.0, borderpad=0.1)

vdf = vdf.rename(columns={'log2_fc': 'log2fc_I_vs_IIb'})
vdf['textbook_side'] = vdf['gene'].map(
    lambda g: 'slow' if g in C.TEXTBOOK_DE_SLOW else
              ('fast' if g in C.TEXTBOOK_DE_FAST else ''))

# ---------------------------------------------------------------------------
# Margin labels for the highlighted markers: slow right, fast left.
# ---------------------------------------------------------------------------
hits = vdf[vdf['highlighted']]


def margin_hits(side):
    sub = hits[hits['textbook_side'] == side]
    return [(r['gene'], float(r['log2fc_I_vs_IIb']), float(r['neg_log10_padj']))
            for _, r in sub.iterrows()]


C.annotate_in_margins(ax, margin_hits('slow'), margin_hits('fast'),
                      color=C.TEXTBOOK_COLOR, fontsize=7, expand=0.30)

plt.tight_layout()
vdf.to_csv(C.DATA / 'volcano_I_vs_IIb.csv', index=False)

C.save_panel(fig, 'panel_f_volcano_I_vs_IIb',
             vdf[['protein_group', 'gene', 'log2fc_I_vs_IIb', 'p_value',
                  'p_adj', 'neg_log10_padj', 'n_a', 'n_b', 'significance',
                  'highlighted', 'textbook_side']])

# ---------------------------------------------------------------------------
# Did every textbook marker land on the side the literature predicts?
# ---------------------------------------------------------------------------
print(f'\nTextbook markers on the volcano '
      f'({len(hits)} of {len(C.TEXTBOOK_DE)} detected):')
rows = []
for _, r in hits.sort_values('log2fc_I_vs_IIb', ascending=False).iterrows():
    observed = 'slow' if r['log2fc_I_vs_IIb'] > 0 else 'fast'
    rows.append({'gene': r['gene'], 'log2FC_I_vs_IIb': round(r['log2fc_I_vs_IIb'], 2),
                 'padj': f"{r['p_adj']:.2e}", 'significance': r['significance'],
                 'expected': r['textbook_side'], 'observed': observed,
                 'agrees': 'yes' if r['textbook_side'] == observed else 'NO'})
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print(f"\nagreeing with the literature: "
      f"{(summary['agrees'] == 'yes').sum()}/{len(summary)}")
missing = sorted(C.TEXTBOOK_DE - set(hits['gene']))
if missing:
    print(f'not detected / filtered out: {missing}')
