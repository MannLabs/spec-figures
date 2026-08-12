"""Supplementary figure 5 — digestion efficiency and membrane-protein coverage."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
OUTDIR = _cfg.output_dir(__file__)

CLASS_ORDER = ['Plasma membrane / surface', 'Mitochondrial membrane',
               'Other membrane', 'Non-membrane']
SHORT = {'Plasma membrane / surface': 'Plasma membrane',
         'Mitochondrial membrane': 'Mitochondrial',
         'Other membrane': 'Other membrane',
         'Non-membrane': 'Non-membrane'}
COLOR = {'Plasma membrane / surface': core.PALETTE_SINGLE[2],   # sky
         'Mitochondrial membrane': core.PALETTE_SINGLE[1],      # lavender
         'Other membrane': core.PALETTE_SINGLE[4],              # yellow
         'Non-membrane': '#bdbdbd'}
MEMBRANE = CLASS_ORDER[:3]

N_DECILES = 10
FONTSIZE = 8
STEM = 'supplement_digestion_and_membrane'

# Fiber-type hues are imported from figure 5 rather than restated, so a fiber type
# keeps one colour across the main figure and this supplement.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'figure5',
                                                'scripts')))
import common as fig5                                            # noqa: E402

DIGESTION_YLIM = (75, 95)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
digestion = pd.read_csv(os.path.join(INPUT, 'per_fiber_digestion.csv'))
wide = pd.read_parquet(os.path.join(INPUT, 'pg_log2_matrix.parquet'))
sample_info = pd.read_parquet(os.path.join(INPUT, 'sample_info_typed.parquet'))
cls = (pd.read_csv(os.path.join(INPUT, 'protein_membrane_class.csv'))
       .set_index('protein_group')['membrane_class'])

table = pd.DataFrame({'median_log2': wide.median(axis=1)})
table['membrane_class'] = cls.reindex(table.index).fillna('Non-membrane')
print(f'{len(table):,} protein groups across {wide.shape[1]} fibers')

counts = table['membrane_class'].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
pct = 100 * counts / counts.sum()
print('\nclass composition:')
for c in CLASS_ORDER:
    print(f'  {c:26s} {counts[c]:>5,}  {pct[c]:5.1f} %')
print(f'  {"membrane total":26s} {counts[MEMBRANE].sum():>5,}  '
      f'{pct[MEMBRANE].sum():5.1f} %')

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
ranked = table.dropna(subset=['median_log2']).copy()
if len(ranked) < len(table):
    print(f'  {len(table) - len(ranked)} group(s) without a median excluded from panel b')
ranked['decile'] = (pd.qcut(ranked['median_log2'], N_DECILES, labels=False) + 1).astype(int)
comp = (ranked.groupby(['decile', 'membrane_class']).size()
        .unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0))
frac = 100 * comp.div(comp.sum(axis=1), axis=0)
print('\nmembrane share per abundance decile (1 = least abundant):')
for dec, row in frac.iterrows():
    print(f'  decile {dec:2d}  membrane {row[MEMBRANE].sum():5.1f} %')
mem_share = frac[MEMBRANE].sum(axis=1)
print(f'  spans {mem_share.min():.1f} to {mem_share.max():.1f} % across the ten deciles')

# ---------------------------------------------------------------------------
# Membrane protein groups quantified per single fiber
# ---------------------------------------------------------------------------
klass = table['membrane_class'].reindex(wide.index)
per_fiber = pd.DataFrame({
    'membrane': wide[klass.isin(MEMBRANE).to_numpy()].notna().sum(axis=0),
    'total': wide.notna().sum(axis=0)})
per_fiber['fiber_type'] = (sample_info.set_index('run')['fiber_type']
                           .reindex(per_fiber.index))
per_fiber['membrane_pct'] = 100 * per_fiber['membrane'] / per_fiber['total']
FIBER_ORDER = [f for f in ['I', 'IIa', 'IIx', 'IIb', 'mixed']
               if f in set(per_fiber['fiber_type'].dropna())]
print('\nmembrane protein groups per single fiber:')
print(per_fiber.groupby('fiber_type')[['membrane', 'membrane_pct']]
      .agg(['median', 'min', 'max']).round(1).reindex(FIBER_ORDER).to_string())

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
digestion = digestion.merge(sample_info[['run', 'fiber_type', 'condition1']],
                            on='run', how='inner')
if len(digestion) != wide.shape[1]:
    raise ValueError(f'{len(digestion)} fibers with a digestion rate against '
                     f'{wide.shape[1]} in the matrix')
print(f'\nfully cleaved precursors, {len(digestion)} fibers:')
print(digestion.groupby('fiber_type')[['mc0_by_count', 'mc0_by_intensity']]
      .agg(['median', 'min', 'max']).round(1).reindex(FIBER_ORDER).to_string())
print(f'  overall median {digestion["mc0_by_count"].median():.1f} % count-weighted, '
      f'{digestion["mc0_by_intensity"].median():.1f} % intensity-weighted')
print(f'  spread across all fibers: {digestion["mc0_by_count"].min():.1f} to '
      f'{digestion["mc0_by_count"].max():.1f} %')

# ---------------------------------------------------------------------------
# Plot — 2 x 2, so every long label has room
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.4))

ax = axes[0, 0]
rng_a = np.random.default_rng(0)
groups_a = [digestion.loc[digestion['fiber_type'] == ft, 'mc0_by_count'].to_numpy()
            for ft in FIBER_ORDER]
bp = ax.boxplot(groups_a, positions=np.arange(len(FIBER_ORDER)), widths=0.62,
                patch_artist=True, showfliers=False, manage_ticks=False,
                medianprops=dict(color='black', linewidth=1.2),
                whiskerprops=dict(color='black', linewidth=0.8),
                capprops=dict(color='black', linewidth=0.8))
for patch, ft in zip(bp['boxes'], FIBER_ORDER):
    patch.set_facecolor(fig5.TYPE_COLOR[ft])
    patch.set_edgecolor('black')
    patch.set_linewidth(0.5)
for i, vals in enumerate(groups_a):
    ax.scatter(i + rng_a.uniform(-0.13, 0.13, len(vals)), vals, s=8,
               color='black', alpha=0.65, linewidth=0, zorder=5)
ax.set_xticks(np.arange(len(FIBER_ORDER)))
ax.set_xticklabels(FIBER_ORDER, fontsize=FONTSIZE)
ax.set_xlim(-0.55, len(FIBER_ORDER) - 0.45)
ax.set_ylim(*DIGESTION_YLIM)
ax.set_xlabel('Fiber type', fontsize=FONTSIZE)
ax.set_ylabel('Fully cleaved precursors [%]', fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# b — protein groups per membrane class
ax = axes[0, 1]
y = np.arange(len(CLASS_ORDER))[::-1]
for yi, c in zip(y, CLASS_ORDER):
    ax.barh(yi, counts[c], height=0.6, color=COLOR[c], edgecolor='black',
            linewidth=0.5, zorder=2)
    ax.text(counts[c] + counts.max() * 0.03, yi, f'{counts[c]:,}  {pct[c]:.0f} %',
            va='center', ha='left', fontsize=6.5)
ax.set_yticks(y)
ax.set_yticklabels([SHORT[c] for c in CLASS_ORDER], fontsize=FONTSIZE)
ax.set_xlim(0, counts.max() * 1.42)
ax.set_xlabel('Protein groups', fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# c — class composition across abundance deciles
ax = axes[1, 0]
bottom = np.zeros(len(frac))
for c in CLASS_ORDER:
    ax.bar(frac.index, frac[c], 0.82, bottom=bottom, color=COLOR[c],
           edgecolor='none', label=SHORT[c], zorder=2)
    bottom += frac[c].to_numpy()
ax.set_xticks(list(frac.index))
ax.set_xticklabels([str(i) for i in frac.index], fontsize=FONTSIZE)
ax.set_xlim(0.4, N_DECILES + 0.6)
ax.set_ylim(0, 100)
ax.set_xlabel('Abundance decile (1 = lowest)', fontsize=FONTSIZE)
ax.set_ylabel('Share of protein groups [%]', fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
          fontsize=6.5, handlelength=0.8, handleheight=0.9, handletextpad=0.35,
          borderpad=0.0, columnspacing=0.9, labelspacing=0.25)

# d — membrane protein groups per fiber
ax = axes[1, 1]
rng = np.random.default_rng(0)
for i, ft in enumerate(FIBER_ORDER):
    vals = per_fiber.loc[per_fiber['fiber_type'] == ft, 'membrane'].to_numpy()
    ax.scatter(np.full(len(vals), i) + rng.uniform(-0.16, 0.16, len(vals)), vals,
               s=13, color=fig5.TYPE_COLOR[ft], alpha=0.8, edgecolor='black',
               linewidth=0.3, zorder=4)
    ax.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color='black',
            linewidth=1.2, zorder=5)
ax.set_xticks(np.arange(len(FIBER_ORDER)))
ax.set_xticklabels(FIBER_ORDER, fontsize=FONTSIZE)
ax.set_xlim(-0.55, len(FIBER_ORDER) - 0.45)
ax.set_ylim(0, None)
ax.set_xlabel('Fiber type', fontsize=FONTSIZE)
ax.set_ylabel('Membrane protein groups\nper fiber', fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
digestion.round(3).to_csv(
    os.path.join(OUTDIR, f'{STEM}_a_sourcedata.csv'), index=False)
pd.DataFrame({'membrane_class': CLASS_ORDER,
              'protein_groups': [counts[c] for c in CLASS_ORDER],
              'percent_of_quantified': [round(pct[c], 2) for c in CLASS_ORDER]}
             ).to_csv(os.path.join(OUTDIR, f'{STEM}_b_sourcedata.csv'), index=False)
frac.round(3).reset_index().to_csv(
    os.path.join(OUTDIR, f'{STEM}_c_sourcedata.csv'), index=False)
per_fiber.reset_index().rename(columns={'index': 'run'}).to_csv(
    os.path.join(OUTDIR, f'{STEM}_d_sourcedata.csv'), index=False)

print(f'\nSaved {STEM} to {OUTDIR}')
