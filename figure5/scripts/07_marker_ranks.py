"""Step 7 — Where the markers sit in the abundance distribution."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import common as C

C.init()

df = C.load_filtered(columns=['run', 'protein_group', 'genes', 'pg_intensity'])
sample_info = C.load_typed()
sel = C.MarkerSelection(df, sample_info)

intensity = sel.pg_wide_all.median(axis=1).dropna().sort_values(ascending=False)
rank = pd.Series(np.arange(1, len(intensity) + 1), index=intensity.index, name='rank')

pd.DataFrame({'rank': rank,
              'gene': sel.pg_to_gene.reindex(intensity.index),
              'median_log2_int': intensity}).to_csv(
    C.DATA / 'intensity_ranks.csv')

print(f'background: {len(intensity):,} quantified protein groups')
print(f'most abundant:  {sel.gene(intensity.index[0]):<12s} log2 = {intensity.iloc[0]:.1f}')
print(f'least abundant: {sel.gene(intensity.index[-1]):<12s} log2 = {intensity.iloc[-1]:.1f}')


def hits(pgs, side):
    rows = [{'protein_group': pg, 'gene': sel.gene(pg), 'rank': int(rank[pg]),
             'median_log2_int': float(intensity[pg]),
             'score': float(sel.score[pg]), 'side': side}
            for pg in pgs if pg in rank.index]
    return pd.DataFrame(rows).sort_values('rank').reset_index(drop=True)


slow_hits = hits(sel.top_slow, 'slow')
fast_hits = hits(sel.top_fast, 'fast')

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
N_LABEL = C.N_TOP    # all 15 per side

fig, ax = plt.subplots(figsize=(4, 4))

ax.scatter(rank.values, intensity.values, s=6, color='black', alpha=0.35,
           edgecolor='none', zorder=1)
for tbl, color in ((slow_hits, C.SLOW_COLOR), (fast_hits, C.FAST_COLOR)):
    ax.scatter(tbl['rank'], tbl['median_log2_int'], s=34, color=color,
               edgecolor='white', linewidth=0.5, zorder=4)

X_SCALE = 'linear'
n_total = len(intensity)
log2_range = intensity.max() - intensity.min()

ax.set_xscale(X_SCALE)
ax.set_xlabel(f'Protein intensity rank '
              f'({"log" if X_SCALE == "log" else "linear"} scale)', fontsize=10)
ax.set_ylabel('Median log₂ protein intensity', fontsize=10)
# Headroom: three label rows above the curve plus the guide captions, three below.
# Widened from 0.34 / 0.52 when LABEL_ROWS went from 2 to 3.
ax.set_ylim(intensity.min() - log2_range * 0.46,
            intensity.max() + log2_range * 0.64)
if X_SCALE == 'log':
    ax.set_xlim(0.7, n_total * 1.4)
else:
    ax.set_xlim(-n_total * 0.04, n_total * 1.04)
C.despine(ax)

LABEL_ROWS = 3


def place_labels(tbl, color, *, above):
    # Strongest first: `tbl` is rank-sorted, so re-sort by |score| and keep the
    # leading N_LABEL, then restore rank order for the left-to-right layout.
    tbl = (tbl.reindex(tbl['score'].abs().sort_values(ascending=False).index)
              .head(N_LABEL).sort_values('rank').reset_index(drop=True))
    n = len(tbl)
    if not n:
        return
    step = log2_range * 0.125
    rows = [(intensity.max() + log2_range * 0.08 + r * step) if above
            else (intensity.min() - log2_range * 0.06 - r * step)
            for r in range(LABEL_ROWS)]

    if X_SCALE == 'log':
        label_xs = 10 ** np.linspace(0, np.log10(n_total), n)
    else:
        label_xs = np.linspace(n_total * 0.01, n_total * 0.99, n)

    for i, r in tbl.iterrows():
        ax.annotate(r['gene'], xy=(r['rank'], r['median_log2_int']),
                    xytext=(label_xs[i], rows[i % LABEL_ROWS]),
                    fontsize=7, fontweight='bold', color=color,
                    ha='center', va='bottom' if above else 'top',
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.4,
                                    alpha=0.6, shrinkA=2, shrinkB=2))


place_labels(slow_hits, C.SLOW_COLOR, above=True)
place_labels(fast_hits, C.FAST_COLOR, above=False)

ax.legend(handles=[
    Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='black',
           markeredgecolor='none', markersize=4, alpha=0.55,
           label=f'All PGs (n={n_total:,})'),
    Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=C.SLOW_COLOR,
           markeredgecolor='none', markersize=5, label=f'Top {C.N_TOP} slow'),
    Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=C.FAST_COLOR,
           markeredgecolor='none', markersize=5, label=f'Top {C.N_TOP} fast'),
], loc='center right', bbox_to_anchor=(1.0, 0.52), fontsize=7, frameon=False,
   labelspacing=0.4, handletextpad=0.3, borderaxespad=0.0)

guide_rows = []
for q, lab in ((0.10, 'top 10%'), (0.50, 'med'), (0.90, 'low 10%')):
    r = int(n_total * q)
    ax.axvline(r, color='#333333', linestyle='--', linewidth=0.9, alpha=0.7,
               zorder=1.5)
    guide_rows.append({'guide': lab, 'rank': r})
    ax.text(r, intensity.min() + log2_range * 0.10, lab, fontsize=6,
            color='#333333', ha='center', va='center', zorder=6,
            bbox=dict(facecolor='white', edgecolor='#333333', linewidth=0.6,
                      boxstyle='round,pad=0.18'))

ax.set_title(f'Marker rank in the intensity distribution\nn={n_total:,} PGs',
             fontsize=11, fontweight='bold')
C.unbold(ax)
plt.tight_layout()

C.save_panel(fig, 'supporting_intensity_rank_markers', {
    'all_protein_groups': pd.DataFrame({
        'protein_group': intensity.index,
        'gene': sel.pg_to_gene.reindex(intensity.index).values,
        'rank': rank.values, 'median_log2_int': intensity.values}),
    'markers': pd.concat([slow_hits, fast_hits], ignore_index=True),
    'rank_guides': pd.DataFrame(guide_rows),
})

both = pd.concat([slow_hits, fast_hits], ignore_index=True)
both['rank_pct'] = (both['rank'] / len(intensity) * 100).round(1)
print('\nRank distribution of the 30 markers:')
print(both[['gene', 'side', 'rank', 'rank_pct', 'median_log2_int']].to_string(index=False))
