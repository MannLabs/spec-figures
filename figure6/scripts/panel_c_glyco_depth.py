"""Figure 6c — glycan compositions per glycoprotein, ISD vs SAX SPEC (LTH69)."""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
OUTDIR = _cfg.output_dir(__file__)

# (display label, file tag) — same order and hues as panels b, e and f.
CONDITIONS = [('ISD', 'InSol'), ('SAX SPEC', 'SPEC')]
LABELS = [label for label, _tag in CONDITIONS]
COLOR = {'ISD': core.PALETTE_SINGLE[1], 'SAX SPEC': core.PALETTE_SINGLE[0]}
SHARED_COLOR = '#9E6E9E'      # the two hues mixed, as in panel f
FONTSIZE = 8
WRITE_RETIRED = False
STEM = 'panel_glyco_site_overlap'


def parse_sites(frame, column):
    """{(accession, position)} from a PEAKS '; '-separated site column."""
    sites = set()
    for accession, cell in zip(frame['Accession'], frame[column].fillna('')):
        for position in str(cell).split(';'):
            position = position.strip()
            if position:
                sites.add((accession, int(position)))
    return sites


def distinct_compositions(cell):
    """Number of distinct glycan compositions in a PEAKS ';'-separated Glycan cell."""
    if pd.isna(cell):
        return 0
    return len({x.strip() for x in str(cell).split(';')
                if x.strip() and x.strip().lower() != 'nan'})


tables, n_sites, o_sites, rows_all, proteins, glycans = {}, {}, {}, {}, {}, {}
for label, tag in CONDITIONS:
    frame = pd.read_csv(os.path.join(INPUT, f'glycan.proteins_{tag}.csv'))
    frame['n_compositions'] = frame['Glycan'].map(distinct_compositions)
    # A glycoprotein is a protein group with >= 1 identified glycan. Without this
    # filter the table's row count is not a glycoprotein count.
    frame['is_glycoprotein'] = frame['#Glycans'].fillna(0) > 0
    tables[label] = frame
    rows_all[label] = set(frame['Accession'].dropna())
    proteins[label] = set(frame.loc[frame['is_glycoprotein'], 'Accession'].dropna())
    n_sites[label] = parse_sites(frame, 'N Glycan Sites')
    o_sites[label] = parse_sites(frame, 'O Glycan Sites')
    glycans[label] = int(frame['n_compositions'].sum())
    print(f'{label:9s} {len(rows_all[label]):4d} protein groups in the run, '
          f'{len(proteins[label]):4d} of them glycoproteins | '
          f'{len(n_sites[label]):4d} N-sites, {len(o_sites[label]):4d} O-sites | '
          f'{glycans[label]:5d} glycan compositions '
          f'({glycans[label] / len(proteins[label]):.1f} per glycoprotein)')

for name, sets in (('glycoproteins', proteins), ('N-glyco sites', n_sites),
                   ('O-glyco sites', o_sites)):
    sol, spec = sets['ISD'], sets['SAX SPEC']
    print(f'{name:15s} in-solution only {len(sol - spec):4d} | '
          f'shared {len(sol & spec):4d} | SPEC only {len(spec - sol):4d}')

only_sol = len(n_sites['ISD'] - n_sites['SAX SPEC'])
shared = len(n_sites['ISD'] & n_sites['SAX SPEC'])
only_spec = len(n_sites['SAX SPEC'] - n_sites['ISD'])

# ---------------------------------------------------------------------------
# Plot — same construction as panel f.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.6, 2.6))
v = venn2(subsets=(only_sol, only_spec, shared), set_labels=LABELS, ax=ax)
for patch_id, condition in (('10', 'ISD'), ('01', 'SAX SPEC')):
    patch = v.get_patch_by_id(patch_id)
    if patch is not None:
        patch.set_color(COLOR[condition])
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)
overlap = v.get_patch_by_id('11')
if overlap is not None:
    overlap.set_color(SHARED_COLOR)
    overlap.set_alpha(0.9)
    overlap.set_edgecolor('black')
    overlap.set_linewidth(0.5)
for text in v.set_labels:
    if text is not None:
        text.set_fontsize(FONTSIZE)
for text in v.subset_labels:
    if text is not None:
        text.set_fontsize(FONTSIZE)
        text.set_text(f'{int(text.get_text()):,}')

fig.tight_layout()
if WRITE_RETIRED:
    fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# Source data — the plotted regions, plus the counts the caption should not omit.
# ---------------------------------------------------------------------------
rows = [{'series': 'venn region', 'quantity': 'N-glycosylation sites',
         'region': 'In solution only', 'value': only_sol},
        {'series': 'venn region', 'quantity': 'N-glycosylation sites',
         'region': 'shared', 'value': shared},
        {'series': 'venn region', 'quantity': 'N-glycosylation sites',
         'region': 'SPEC only', 'value': only_spec}]
for label, _tag in CONDITIONS:
    rows += [
        {'series': 'condition total', 'quantity': 'protein groups in run',
         'region': label, 'value': len(rows_all[label])},
        {'series': 'condition total',
         'quantity': 'glycoproteins (>=1 glycan)',
         'region': label, 'value': len(proteins[label])},
        {'series': 'condition total', 'quantity': 'N-glycosylation sites',
         'region': label, 'value': len(n_sites[label])},
        {'series': 'condition total', 'quantity': 'O-glycosylation sites',
         'region': label, 'value': len(o_sites[label])},
        {'series': 'condition total',
         'quantity': 'distinct glycan compositions, summed over glycoproteins',
         'region': label, 'value': glycans[label]},
    ]
if WRITE_RETIRED:
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'),
                          index=False)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
STEM_DEPTH = 'panel_c_glyco_depth'
indexed = {label: tables[label].set_index('Accession') for label in LABELS}
shared_proteins = sorted(proteins['SAX SPEC'] & proteins['ISD'])
depth = pd.DataFrame({
    label: indexed[label].loc[shared_proteins, 'n_compositions']
    for label in LABELS})
if (depth <= 0).any().any():
    raise ValueError('a shared glycoprotein has no glycan composition')
n_up = int((depth['SAX SPEC'] > depth['ISD']).sum())
n_eq = int((depth['SAX SPEC'] == depth['ISD']).sum())
n_down = int((depth['SAX SPEC'] < depth['ISD']).sum())
print(f'\nglycan compositions on the {len(depth)} shared glycoproteins: '
      f'median SAX SPEC {depth["SAX SPEC"].median():.0f} vs '
      f'ISD {depth["ISD"].median():.0f}')
print(f'  SPEC higher on {n_up}, equal on {n_eq}, lower on {n_down}')

fig, ax = plt.subplots(figsize=(3.0, 3.0))
low = 0.8 * depth.min().min()
limit = 1.3 * depth.max().max()
ax.plot([low, limit], [low, limit], color='black', linestyle='--',
        linewidth=0.7, zorder=2)
ax.scatter(depth['ISD'], depth['SAX SPEC'], s=16, color=COLOR['SAX SPEC'],
           alpha=0.65, edgecolor='black', linewidth=0.3, zorder=4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(low, limit)
ax.set_ylim(low, limit)
ax.set_aspect('equal')
ax.text(0.04, 0.96,
        f'n = {len(depth)} glycoproteins\nmedian '
        f'{depth["SAX SPEC"].median():.0f} vs {depth["ISD"].median():.0f}',
        transform=ax.transAxes, fontsize=FONTSIZE, va='top', ha='left')
# The unit goes in the axis labels, not only into the caption: the count is per
# PROTEIN GROUP, and it is distinct glycan compositions, not glycopeptides.
ax.set_xlabel('Glycan compositions\nper protein group, ISD', fontsize=FONTSIZE)
ax.set_ylabel('Glycan compositions\nper protein group, SAX SPEC',
              fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
core.set_axes_size_inches(fig, ax, w_in=1.85, h_in=1.85)
fig.savefig(os.path.join(OUTDIR, f'{STEM_DEPTH}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM_DEPTH}.png'), dpi=300,
            bbox_inches='tight')

depth.reset_index().rename(
    columns={'index': 'accession', 'Accession': 'accession'}).assign(
    series='glycoprotein detected in both',
    quantity='distinct glycan compositions per protein group').to_csv(
    os.path.join(OUTDIR, f'{STEM_DEPTH}_sourcedata.csv'), index=False)

print(f'\nSaved {STEM} and {STEM_DEPTH} to {OUTDIR}')
