"""Figure 3g + 3h — sequence coverage and peptides per protein, single elution vs"""

import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
# One search per arm, as in panels e and f.
REPORTS = {'Single elution': os.path.join(INPUT, 'LTH43', 'single_elution',
                                          'report.parquet'),
           'On-SPEC fractionation': os.path.join(INPUT, 'LTH43', 'fractionated',
                                                 'report.parquet')}
FASTA = os.path.join(INPUT, 'HYA.fasta')
OUTDIR = _cfg.output_dir(__file__)
os.makedirs(OUTDIR, exist_ok=True)

QVALUE = 0.01
PAIRED = True
FRACTION_ROWS = ['A', 'B', 'C', 'D', 'E']
SPECIES = ['HUMAN', 'ARATH', 'YEAST']
SPECIES_LABEL = {'HUMAN': 'Human', 'ARATH': 'Arabidopsis', 'YEAST': 'Yeast'}
SPECIES_COLOR = {'HUMAN': core.PALETTE_SINGLE[1], 'ARATH': core.PALETTE_SINGLE[4],
                 'YEAST': core.PALETTE_SINGLE[3]}
CONDITIONS = [('On-SPEC fractionation', '-'), ('Single elution', '--')]
XMAX_PEPTIDES = 50

# ---------------------------------------------------------------------------
# FASTA: accession -> (sequence, species)
# ---------------------------------------------------------------------------
sequences, species_of = {}, {}
acc, buf = None, []
with open(FASTA, encoding='utf-8', errors='replace') as fh:
    for line in fh:
        if line.startswith('>'):
            if acc:
                sequences[acc] = ''.join(buf)
            m = re.match(r'>(?:sp|tr)\|([^|]+)\|', line) or re.match(r'>(\S+)', line)
            acc, buf = m.group(1), []
            sp = re.search(r'_(HUMAN|ARATH|YEAST)', line)
            species_of[acc] = sp.group(1) if sp else None
        else:
            buf.append(line.strip())
if acc:
    sequences[acc] = ''.join(buf)
print(f'FASTA: {len(sequences):,} entries')

# ---------------------------------------------------------------------------
# Peptides per protein group, per condition.
# ---------------------------------------------------------------------------
frames = []
for condition, path in REPORTS.items():
    frames.append(pd.read_parquet(path, columns=[
        'Run', 'Protein.Group', 'Stripped.Sequence', 'Decoy', 'Q.Value',
        'PG.Q.Value']).assign(condition=condition))
d = pd.concat(frames, ignore_index=True)
d = d[(d['Decoy'] == 0) & (d['Q.Value'] < QVALUE) & (d['PG.Q.Value'] < QVALUE)]
d = d[d['Protein.Group'].astype(str) != ''].copy()
print('runs per condition:')
print(d.groupby('condition')['Run'].nunique().to_string())


def coverage_percent(group, peptides):
    """Covered residues of the group's leading accession, in percent."""
    members = [a for a in str(group).split(';') if a in sequences]
    if not members:
        return np.nan, len(peptides)
    seq = sequences[members[0]]
    covered = np.zeros(len(seq), dtype=bool)
    unmatched = 0
    seq_il = seq.replace('L', 'I')
    for pep in peptides:
        hits = 0
        for target in (seq, seq_il):
            probe = pep if target is seq else pep.replace('L', 'I')
            start = target.find(probe)
            while start != -1:
                covered[start:start + len(probe)] = True
                hits += 1
                start = target.find(probe, start + 1)
            if hits:
                break
        if not hits:
            # Peptide belongs to another member of the group.
            if not any(pep in sequences[a] for a in members[1:]):
                unmatched += 1
    return 100 * covered.mean(), unmatched


rows, unmatched_total, peptide_total = [], 0, 0
for (cond, group), grp in d.groupby(['condition', 'Protein.Group'], sort=False):
    peptides = set(grp['Stripped.Sequence'])
    cov, unmatched = coverage_percent(group, peptides)
    lead = str(group).split(';')[0]
    rows.append({'condition': cond, 'protein_group': group,
                 'species': species_of.get(lead), 'n_peptides': len(peptides),
                 'coverage_pct': cov})
    unmatched_total += unmatched
    peptide_total += len(peptides)

per_protein = pd.DataFrame(rows).dropna(subset=['species', 'coverage_pct'])
print(f'\npeptides not locatable in their group: {unmatched_total:,} of '
      f'{peptide_total:,} ({100 * unmatched_total / peptide_total:.2f}%)')

def report(frame, title):
    summary = (frame.groupby(['condition', 'species'])
               .agg(n_proteins=('protein_group', 'size'),
                    median_coverage=('coverage_pct', 'median'),
                    median_peptides=('n_peptides', 'median'))
               .reindex([(c, s) for c, _ls in CONDITIONS for s in SPECIES]))
    print(f'\n{title}:')
    print(summary.round(1).to_string())


report(per_protein, 'all protein groups of each condition')

# Protein groups seen in both conditions, and the per-protein change.
shared = (per_protein.groupby('protein_group')['condition'].nunique()
          .loc[lambda s: s == 2].index)
paired = per_protein[per_protein['protein_group'].isin(shared)]
report(paired, f'paired subset — {len(shared):,} protein groups in both conditions')

wide = paired.pivot_table(index=['species', 'protein_group'], columns='condition',
                          values=['coverage_pct', 'n_peptides'])
for sp, grp in wide.groupby(level=0):
    dc = (grp[('coverage_pct', 'On-SPEC fractionation')]
          - grp[('coverage_pct', 'Single elution')])
    dp = (grp[('n_peptides', 'On-SPEC fractionation')]
          - grp[('n_peptides', 'Single elution')])
    print(f'  {sp:6s} median change: coverage {dc.median():+.1f} points '
          f'({100 * (dc > 0).mean():.0f}% of proteins improve), '
          f'peptides {dp.median():+.0f}')

only_frac = per_protein[(per_protein['condition'] == 'On-SPEC fractionation')
                        & ~per_protein['protein_group'].isin(shared)]
print('\nprotein groups detected only with fractionation:')
print(only_frac.groupby('species')
      .agg(n=('protein_group', 'size'), median_coverage=('coverage_pct', 'median'),
           median_peptides=('n_peptides', 'median')).round(1).to_string())

if PAIRED:
    per_protein = paired

# ---------------------------------------------------------------------------
# Plot — two ECDFs, same construction; g carries the linestyle key for both.
# ---------------------------------------------------------------------------
def draw_ecdf(column, xlabel, xmax, filename, legend, median_fmt):
    fig, ax = plt.subplots(figsize=(3.7, 2))
    for cond, ls in CONDITIONS:
        for sp in SPECIES:
            v = np.sort(per_protein.loc[(per_protein['condition'] == cond)
                                        & (per_protein['species'] == sp),
                                        column].to_numpy())
            if not len(v):
                continue
            y = np.arange(1, len(v) + 1) / len(v)
            ax.step(v, y, ls, where='post', color=SPECIES_COLOR[sp],
                    linewidth=1.3, zorder=3)

    # Species entries carry the single-elution -> fractionation medians, so the
    # legend doubles as the numeric readout; the linestyle key is on g only.
    def median(cond, sp):
        return per_protein.loc[(per_protein['condition'] == cond)
                               & (per_protein['species'] == sp), column].median()

    handles = [Line2D([], [], color=SPECIES_COLOR[sp], linewidth=1.6,
                      label=f'{SPECIES_LABEL[sp]} '
                            f'({median("Single elution", sp):{median_fmt}}'
                            f'$\\to${median("On-SPEC fractionation", sp):{median_fmt}})')
               for sp in SPECIES]
    if legend:
        handles += [Line2D([], [], color='black', linewidth=1.3, linestyle=ls,
                           label=cond) for cond, ls in CONDITIONS]
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=7,
              handlelength=1.4, labelspacing=0.25, borderpad=0.2)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel('Cumulative fraction', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{filename}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{filename}.png'), dpi=300, bbox_inches='tight')

    src = per_protein.assign(
        species=per_protein['species'].map(SPECIES_LABEL)).sort_values(
        ['condition', 'species', column])
    src['cumulative_fraction'] = (src.groupby(['condition', 'species'])[column]
                                  .transform(lambda s: np.arange(1, len(s) + 1) / len(s)))
    src[['condition', 'species', 'protein_group', 'n_peptides', 'coverage_pct',
         'cumulative_fraction']].to_csv(
        os.path.join(OUTDIR, f'{filename}_sourcedata.csv'), index=False)


draw_ecdf('coverage_pct', 'Protein sequence coverage [%]', 100,
          'panel_g_sequence_coverage', legend=True, median_fmt='.0f')
draw_ecdf('n_peptides', 'Peptides per protein', XMAX_PEPTIDES,
          'panel_h_peptides_per_protein', legend=False, median_fmt='.0f')

print(f'\nSaved panels g and h to {OUTDIR}')
