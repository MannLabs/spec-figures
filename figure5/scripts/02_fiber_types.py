"""Step 2 — Per-fiber MyHC composition from precursors UNIQUELY attributable to"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

C.init()

TYPE_THRESHOLDS = {'I': 0.70, 'IIa': 0.60, 'IIx': 0.60, 'IIb': 0.70}

# ---------------------------------------------------------------------------
# 1. precursor -> MyHC isoform map, from the raw report
# ---------------------------------------------------------------------------
raw = pd.read_parquet(C.REPORT,
                      columns=['Precursor.Id', 'Protein.Ids', 'PG.Q.Value'])
raw = raw[raw['PG.Q.Value'] < 0.01]
pid_to_acc = raw.drop_duplicates('Precursor.Id').set_index('Precursor.Id')['Protein.Ids']


def assign_myh(protein_ids):
    hits = {C.ACC_TO_MYH[a] for a in protein_ids.split(';') if a in C.ACC_TO_MYH}
    return hits.pop() if len(hits) == 1 else None


precursor_myh = pid_to_acc.map(assign_myh).dropna()
print('Isoform-unique precursors:')
print(precursor_myh.value_counts().to_string())
precursor_myh.rename('myh').to_csv(C.DATA / 'myh_unique_precursors.csv')

# ---------------------------------------------------------------------------
# 2. per-fiber MyHC abundance
# ---------------------------------------------------------------------------
df = C.load_filtered(columns=['run', 'precursor_id', 'precursor_intensity'])
sample_info = pd.read_parquet(C.SI_FILTERED)

df = df[df['precursor_id'].isin(precursor_myh.index)].copy()
df['myh'] = df['precursor_id'].map(precursor_myh)

myh_per_fiber = (df.groupby(['run', 'myh'])['precursor_intensity']
                   .sum().unstack(fill_value=0.0)
                   .reindex(columns=C.MYH_ORDER, fill_value=0.0)
                   .reindex(sample_info['run']).fillna(0.0))
myh_per_fiber.columns = [C.MYH_TO_TYPE[g] for g in myh_per_fiber.columns]

totals = myh_per_fiber.sum(axis=1)
fractions = myh_per_fiber.div(totals.where(totals > 0, np.nan), axis=0)

# ---------------------------------------------------------------------------
# 3. classify
# ---------------------------------------------------------------------------
def classify(row):
    if row.isna().all():
        return 'no_id'
    for t, thr in TYPE_THRESHOLDS.items():
        if row.get(t, 0) >= thr:
            return t
    return 'mixed'


fiber_type = fractions.apply(classify, axis=1).rename('fiber_type')
dominant = fractions.idxmax(axis=1).rename('dominant_myhc')
fast_slow = dominant.map(
    lambda t: 'slow' if t == 'I' else ('fast_IIa' if t == 'IIa' else 'fast_IIxb')
).rename('fast_slow')

out = sample_info.copy()
for frame in (myh_per_fiber.add_prefix('int_'), fractions.add_prefix('frac_'),
              fiber_type, dominant, fast_slow):
    out = out.merge(frame.reset_index(), on='run', how='left')
out.to_parquet(C.SI_TYPED)
out.to_csv(C.DATA / 'fiber_myhc.csv', index=False)

print('\nFiber type counts:')
print(out['fiber_type'].value_counts().to_string())
print('\nFiber type x muscle:')
print(pd.crosstab(out['fiber_type'], out['condition1']).to_string())
print('\nfast_slow x muscle:')
print(pd.crosstab(out['fast_slow'], out['condition1']).to_string())

# ---------------------------------------------------------------------------
# 4a. Per-fiber composition strip (3 horizontal slots wide)
# ---------------------------------------------------------------------------
order = fractions.copy()
order['_score'] = (1.00 * order['I'].fillna(0) + 0.66 * order['IIa'].fillna(0)
                   + 0.33 * order['IIx'].fillna(0))
order = order.sort_values('_score', ascending=False).index

fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True,
                         gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.05})

muscle = sample_info.set_index('run').loc[order, 'condition1']
ax_top = axes[0]
ax_top.bar(range(len(order)), [1] * len(order),
           color=[C.MUSCLE_COLOR[m] for m in muscle], width=1.0,
           edgecolor='none')
ax_top.set_yticks([])
ax_top.set_ylabel('Muscle', rotation=0, ha='right', va='center', fontsize=9)
ax_top.set_xlim(-0.5, len(order) - 0.5)
for s in ('top', 'right', 'left', 'bottom'):
    ax_top.spines[s].set_visible(False)
ax_top.legend([plt.Rectangle((0, 0), 1, 1, color=C.MUSCLE_COLOR[m])
               for m in C.MUSCLE_ORDER], C.MUSCLE_ORDER,
              loc='center left', bbox_to_anchor=(1.005, 0.5), frameon=False,
              fontsize=8, title='Muscle')

ax = axes[1]
fracs_ordered = fractions.loc[order].fillna(0)
x = np.arange(len(order))
bottom = np.zeros(len(order))
for t in C.TYPE_ORDER:
    ax.bar(x, fracs_ordered[t], bottom=bottom, color=C.TYPE_COLOR[t],
           label=f'{C.MYH_GENE[t]} ({t})', width=1.0, edgecolor='none')
    bottom += fracs_ordered[t].values
ax.set_ylim(0, 1)
ax.set_xlim(-0.5, len(order) - 0.5)
ax.set_ylabel('MyHC fraction (per fiber)')
ax.set_xticks([])
ax.set_xlabel(f'Fibers (n={len(order)}), sorted slow → fast')
ax.legend(loc='center left', bbox_to_anchor=(1.005, 0.5), frameon=False,
          fontsize=9, title='MyHC isoform')

typed = out.set_index('run').loc[order, 'fiber_type']
for i0, i1, t in C.type_spans(typed.tolist(), min_n=3):
    if t == 'mixed':
        continue
    ax.text((i0 + i1 - 1) / 2.0, 1.02, f'{t}\nn={i1 - i0}',
            ha='center', va='bottom', fontsize=8, color='black')

fig.suptitle('Per-fiber MyHC composition · isoform-unique precursors · '
             'Murgia-style classification', fontsize=11, y=0.99)
C.unbold(*fig.axes)

sd_stack = (fracs_ordered.reset_index()
            .melt(id_vars='run', var_name='fiber_type', value_name='myhc_fraction'))
sd_stack['fiber_index'] = sd_stack['run'].map(
    {r: i for i, r in enumerate(order)})
C.save_panel(fig, 'supporting_myhc_stack', {
    'myhc_fraction': sd_stack,
    'muscle': pd.DataFrame({'run': list(order),
                            'fiber_index': range(len(order)),
                            'muscle': muscle.values,
                            'fiber_type': typed.values}),
})

# ---------------------------------------------------------------------------
# 4b. Per-isoform fraction by muscle — 4 horizontal slots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
rng = np.random.default_rng(0)
rows = []
for ax, t in zip(axes, C.TYPE_ORDER):
    for i, m in enumerate(C.MUSCLE_ORDER):
        runs_m = out.loc[out['condition1'] == m, 'run']
        vals = fractions.reindex(runs_m)[t].dropna()
        ax.scatter(i + rng.uniform(-0.15, 0.15, size=len(vals)), vals,
                   s=18, color=C.MUSCLE_COLOR[m], alpha=0.7, edgecolor='none')
        rows.append(pd.DataFrame({'isoform': f'{C.MYH_GENE[t]} ({t})',
                                  'muscle': m, 'run': vals.index,
                                  'fraction': vals.values}))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(C.MUSCLE_ORDER)
    ax.set_title(f'{C.MYH_GENE[t]} ({t})', color=C.TYPE_COLOR[t])
    ax.set_ylim(-0.05, 1.05)
    C.despine(ax)
axes[0].set_ylabel('MyHC fraction')
C.unbold(*axes)
plt.tight_layout()
C.save_panel(fig, 'supporting_myhc_fraction_by_muscle',
             pd.concat(rows, ignore_index=True))
