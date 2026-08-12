"""Figure 6b + 6d — plasma glycoproteomics with SPEC (LTH69)."""

import os
import sys
import re

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

CONDITIONS = ['In solution', 'SPEC']
LABEL = {'In solution': 'ISD', 'SPEC': 'SAX SPEC'}
# SPEC keeps its coral; the in-solution digest takes the ISD lavender of figure 2.
COLOR = {'In solution': core.PALETTE_SINGLE[1], 'SPEC': core.PALETTE_SINGLE[0]}
GLYCO_COLOR = core.PALETTE_SINGLE[0]
WRITE_RETIRED = False

ATLAS_CONC = 'Blood concentration - Conc. blood MS [pg/L]'
FONTSIZE = 8
BAR_WIDTH_IN = 0.38
SHOW_VALUE_LABELS = False
POINT_SIZE = core.replicate_point_size(BAR_WIDTH_IN)


def glycopeptides_per_sample(path):
    """Per-sample #Glycopeptides from the T1 block of a PEAKS glycan summary."""
    lines = open(path, encoding='utf-8-sig').read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('T1.'))
    header = next(l for l in lines[start:start + 4] if '#Glycopeptides' in l)
    column = header.split(',').index('#Glycopeptides')
    counts = {}
    for line in lines[start + 1:]:
        # The T2 block repeats the "Sample" rows with a narrower layout, so the
        # parse has to stop at the end of T1.
        if re.match(r'^T\d+\.', line) or (counts and not line.strip()):
            break
        if not line.startswith('Sample'):
            continue
        fields = line.split(',')
        counts[fields[0]] = int(fields[column])
    return counts


# ---------------------------------------------------------------------------
# b — glycopeptide identifications
# ---------------------------------------------------------------------------
rows = []
for condition, fname in [('In solution', 'glycan.summary-table_InSol.csv'),
                         ('SPEC', 'glycan.summary-table_SPEC.csv')]:
    for replicate, (sample, value) in enumerate(
            sorted(glycopeptides_per_sample(os.path.join(INPUT, fname)).items(),
                   key=lambda kv: int(kv[0].split()[1])), start=1):
        rows.append({'condition': condition, 'sample': sample,
                     'replicate': replicate, 'glycopeptides': value})
counts = pd.DataFrame(rows)
means = counts.groupby('condition')['glycopeptides'].mean().reindex(CONDITIONS)
print('glycopeptides per run:')
print(counts.pivot(index='replicate', columns='condition',
                   values='glycopeptides')[CONDITIONS].to_string())
gain = 100 * (means['SPEC'] / means['In solution'] - 1)
print(f'\nmeans: {means["In solution"]:,.0f} -> {means["SPEC"]:,.0f}  ({gain:+.1f}%)')

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(2.2, 2.6))
for i, condition in enumerate(CONDITIONS):
    vals = counts.loc[counts['condition'] == condition, 'glycopeptides'].to_numpy()
    ax.bar(i, vals.mean(), 0.6, color=COLOR[condition], edgecolor='black',
           linewidth=0.5, zorder=2)
    ax.scatter(np.full(len(vals), i) + rng.uniform(-0.09, 0.09, len(vals)), vals,
               s=POINT_SIZE, color='black', alpha=0.75, linewidth=0.3, edgecolor='white',
               zorder=5)
    if SHOW_VALUE_LABELS:
        ax.text(i, vals.max() * 1.03, f'{vals.mean():,.0f}', ha='center',
                va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels([LABEL[c] for c in CONDITIONS], fontsize=FONTSIZE,
                   rotation=45, ha='right')
ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
ax.set_ylim(0, counts['glycopeptides'].max() * 1.18)
ax.set_ylabel('Glycopeptide identifications', fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.canvas.draw()          # fixed drawn bar width, as in figures 2-4
ax_w = ax.get_position().width * fig.get_size_inches()[0]
w_data = BAR_WIDTH_IN * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax_w
for patch in ax.patches:
    centre = patch.get_x() + patch.get_width() / 2
    patch.set_width(w_data)
    patch.set_x(centre - w_data / 2)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'panel_b_glycopeptides.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_b_glycopeptides.png'), dpi=300,
            bbox_inches='tight')
counts.to_csv(os.path.join(OUTDIR, 'panel_b_glycopeptides_sourcedata.csv'), index=False)

# ---------------------------------------------------------------------------
# c — glycoproteins across the plasma concentration range
# ---------------------------------------------------------------------------
atlas = pd.read_csv(os.path.join(INPUT, 'proteinatlas.tsv'), sep='\t',
                    low_memory=False)
atlas = atlas[['Gene', 'Uniprot', ATLAS_CONC]].copy()
atlas[ATLAS_CONC] = pd.to_numeric(atlas[ATLAS_CONC], errors='coerce')
atlas = (atlas[atlas[ATLAS_CONC] > 0]
         .sort_values(ATLAS_CONC, ascending=False).reset_index(drop=True))

glyco = pd.read_csv(os.path.join(INPUT, 'glycan.proteins_SPEC.csv'))
glyco['accession'] = glyco['Accession'].astype(str).str.split('|').str[0]
sites_n = pd.to_numeric(glyco['#N Glycan Sites'], errors='coerce').fillna(0)
sites_o = pd.to_numeric(glyco['#O Glycan Sites'], errors='coerce').fillna(0)
glyco['has_glycan'] = (sites_n + sites_o) > 0
print(f'\nglycan protein export: {len(glyco)} proteins, '
      f'{int(glyco["has_glycan"].sum())} with a detected glycan site')
accessions = set(glyco.loc[glyco['has_glycan'], 'accession'])
atlas['detected'] = atlas['Uniprot'].astype(str).str.split(', ').map(
    lambda lst: any(a.strip() in accessions for a in lst))

# The published panel stops at the lowest-ranked detected glycoprotein; keeping the
# full 4,285-protein ranking would add 2,450 grey points below the last coral one.
last = int(np.flatnonzero(atlas['detected'].to_numpy())[-1])
plot_df = atlas.iloc[:last + 1].copy()
plot_df['rank'] = np.arange(1, len(plot_df) + 1)
plot_df['log10_concentration'] = np.log10(plot_df[ATLAS_CONC])
print(f'\nplasma ranking: {len(atlas):,} proteins with a blood MS concentration, '
      f'truncated at the lowest detected glycoprotein (rank {last + 1})')
print(f'detected glycoproteins in the plotted range: '
      f'{int(plot_df["detected"].sum())} of {len(accessions)} '
      f'({100 * plot_df["detected"].sum() / len(plot_df):.1f}% of the ranking)')
print('log10 concentration range: '
      f'{plot_df["log10_concentration"].max():.2f} to {plot_df["log10_concentration"].min():.2f}')

fig, ax = plt.subplots(figsize=(4.0, 2.6))
rest = plot_df[~plot_df['detected']]
hit = plot_df[plot_df['detected']]
ax.scatter(rest['rank'], rest['log10_concentration'], s=6, color='#CCCCCC',
           edgecolor='none', label='No glycosylation detected', zorder=2)
ax.scatter(hit['rank'], hit['log10_concentration'], s=14, color=GLYCO_COLOR,
           edgecolor='black', linewidth=0.3, label='Detected glycoprotein', zorder=4)

ax.set_xlabel('Protein rank by plasma concentration', fontsize=FONTSIZE)
ax.set_ylabel('log₁₀ concentration [pg/L]', fontsize=FONTSIZE)
ax.set_xlim(0, len(plot_df) * 1.02)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', frameon=False, fontsize=7, handletextpad=0.3,
          borderpad=0.2, labelspacing=0.3)

fig.tight_layout()
if WRITE_RETIRED:
    fig.savefig(os.path.join(OUTDIR, 'panel_glycoprotein_rank.pdf'),
                bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, 'panel_glycoprotein_rank.png'), dpi=300,
                bbox_inches='tight')
    plot_df[['rank', 'Gene', 'Uniprot', ATLAS_CONC, 'log10_concentration',
             'detected']].to_csv(
        os.path.join(OUTDIR, 'panel_glycoprotein_rank_sourcedata.csv'), index=False)
plt.close(fig)

print(f'\nSaved panels b and c to {OUTDIR}')


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
GROUPS = [('Not detected', '#BFBFBF'),
          ('SAX SPEC', core.PALETTE_SINGLE[0]),
          ('ISD', core.PALETTE_SINGLE[1])]

glyco_sets = {}
for label, tag in (('SAX SPEC', 'SPEC'), ('ISD', 'InSol')):
    g = pd.read_csv(os.path.join(INPUT, f'glycan.proteins_{tag}.csv'))
    n = pd.to_numeric(g['#N Glycan Sites'], errors='coerce').fillna(0)
    o = pd.to_numeric(g['#O Glycan Sites'], errors='coerce').fillna(0)
    glyco_sets[label] = set(g.loc[(n + o) > 0, 'Accession']
                            .astype(str).str.split('|').str[0])

abundance = atlas.copy()
abundance['accession'] = (abundance['Uniprot'].astype(str)
                          .str.split(',').str[0].str.strip())
abundance['log10_concentration'] = np.log10(abundance[ATLAS_CONC])
abundance['in_spec'] = abundance['accession'].isin(glyco_sets['SAX SPEC'])
abundance['in_isd'] = abundance['accession'].isin(glyco_sets['ISD'])

series = {'Not detected': abundance.loc[~(abundance['in_spec'] | abundance['in_isd']),
                                       'log10_concentration'],
          'SAX SPEC': abundance.loc[abundance['in_spec'], 'log10_concentration'],
          'ISD': abundance.loc[abundance['in_isd'], 'log10_concentration']}
print('\nplasma proteins with an HPA MS concentration:', len(abundance))
for label, _color in GROUPS:
    print(f'  {label:13s} n = {len(series[label]):5d}  '
          f'median log10 = {series[label].median():.2f}')

# The exclusive sets are what would explain a count difference, so they are
# reported even though they are too small (13 and 6 mapped) to draw as a histogram.
only_spec = abundance[abundance['in_spec'] & ~abundance['in_isd']]
only_isd = abundance[abundance['in_isd'] & ~abundance['in_spec']]
print(f'  SAX SPEC only n = {len(only_spec):3d}  median log10 = '
      f'{only_spec["log10_concentration"].median():.2f}  '
      f'floor {only_spec["log10_concentration"].min():.2f}')
print(f'  ISD only      n = {len(only_isd):3d}  median log10 = '
      f'{only_isd["log10_concentration"].median():.2f}  '
      f'floor {only_isd["log10_concentration"].min():.2f}')

bins = np.arange(np.floor(abundance['log10_concentration'].min() * 2) / 2,
                 abundance['log10_concentration'].max() + 0.5, 0.5)
fig, ax = plt.subplots(figsize=(6.2, 2.0))
for label, color in GROUPS:
    ax.hist(series[label], bins=bins, weights=np.full(len(series[label]),
                                                      1 / len(series[label])),
            histtype='stepfilled', color=color, alpha=0.45, edgecolor=color,
            linewidth=1.2, label=f'{label} (n = {len(series[label]):,})',
            zorder=2)

ax.set_xlabel('log₁₀ Human Protein Atlas plasma concentration [pg/L]',
              fontsize=FONTSIZE)
ax.set_ylabel('Fraction of group', fontsize=FONTSIZE)
ax.set_xlim(bins[0], bins[-1])
ax.set_ylim(0, None)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', frameon=False, fontsize=FONTSIZE, handlelength=1.0,
          handleheight=0.9, handletextpad=0.4, borderpad=0.2, labelspacing=0.3)

fig.tight_layout()
core.set_axes_size_inches(fig, ax, w_in=5.30, h_in=1.30)
fig.savefig(os.path.join(OUTDIR, 'panel_d_glyco_abundance.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_d_glyco_abundance.png'), dpi=300,
            bbox_inches='tight')

out = abundance[['Gene', 'accession', ATLAS_CONC, 'log10_concentration',
                 'in_spec', 'in_isd']].copy()
out['group'] = np.where(out['in_spec'] & out['in_isd'], 'both',
                        np.where(out['in_spec'], 'SAX SPEC only',
                                 np.where(out['in_isd'], 'ISD only',
                                          'not detected')))
out.to_csv(os.path.join(OUTDIR, 'panel_d_glyco_abundance_sourcedata.csv'),
           index=False)

print(f'Saved the abundance histogram to {OUTDIR}')
