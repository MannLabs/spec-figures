"""Supplementary figure 1 — what kind of proteins each phase captures."""

import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from pyteomics import electrochem
from scipy.stats import gaussian_kde

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
BASE = os.path.join(INPUT, 'H032_E297')
FASTA = os.path.join(INPUT, 'fasta', 'Human.fasta')
OUTDIR = _cfg.output_dir(__file__)

# (label, folder, parquet stem, format, hue) — the ISD+ folder holds
# ISD_SDB-RPS.* files. Order and hues match figure 1c.
CONDITIONS = [
    ('C18',  'C18',  'C18',         'on-tip',      core.PALETTE_SINGLE[3]),
    ('ISD+', 'ISD+', 'ISD_SDB-RPS', 'in-solution', core.PALETTE_SINGLE[5]),
    ('SCX',  'SCX',  'SCX',         'on-tip',      core.PALETTE_SINGLE[2]),
    ('ISD',  'ISD',  'ISD',         'in-solution', core.PALETTE_SINGLE[1]),
    ('SAX',  'SAX',  'SAX',         'on-tip',      core.PALETTE_SINGLE[0]),
]
ORDER = [c[0] for c in CONDITIONS]
COLOR = {c[0]: c[4] for c in CONDITIONS}
FORMAT = {c[0]: c[3] for c in CONDITIONS}

QVALUE = 0.01
N_REPLICATES = 4
GRID = 400
FONTSIZE = 8
BASIC_PI = 8.0
AA = set('ACDEFGHIKLMNPQRSTVWY')
STEM = 'supplement_phase_properties_protein_groups'
LENGTH_TICKS = [100, 300, 1000, 3000]

PROPERTIES = [
    {'key': 'gravy', 'label': 'GRAVY (Kyte-Doolittle)', 'ref': 0.0,
     'readout': 'median', 'fmt': '{:+.2f}', 'log': False,
     'legend_title': 'weighted median'},
    {'key': 'pi', 'label': 'Isoelectric point', 'ref': BASIC_PI,
     'readout': 'basic_fraction', 'fmt': '{:.1%}', 'log': False,
     'legend_title': f'% of signal at pI > {BASIC_PI:.0f}'},
    {'key': 'length', 'label': 'Protein length (residues)', 'ref': None,
     'readout': 'median', 'fmt': '{:,.0f}', 'log': True,
     'legend_title': 'weighted median'},
]


def read_fasta(path):
    seqs, acc, buf = {}, None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith('>'):
                if acc is not None:
                    seqs[acc] = ''.join(buf)
                m = re.match(r'>\w+\|([^|]+)\|', line.strip())
                acc = m.group(1) if m else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if acc is not None:
        seqs[acc] = ''.join(buf)
    return seqs


def properties_of(sequence):
    cs = ''.join(ch for ch in sequence if ch in AA)
    if not cs:
        return np.nan, np.nan, np.nan
    return core.gravy(cs), electrochem.pI(cs), np.log10(len(cs))


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(np.interp(q, (np.cumsum(w) - 0.5 * w) / w.sum(), v))


def legend_value(prop, values, weights):
    if prop['readout'] == 'median':
        med = weighted_quantile(values, weights, 0.5)
        return prop['fmt'].format(10.0 ** med if prop['log'] else med)
    return prop['fmt'].format(weights[values > BASIC_PI].sum() / weights.sum())


sequences = read_fasta(FASTA)
print(f'FASTA entries: {len(sequences):,}')

# ---------------------------------------------------------------------------
# Per condition: one row per protein group with its properties and weight.
# ---------------------------------------------------------------------------
cache, data = {}, {}
for label, folder, stem, _fmt, _color in CONDITIONS:
    d = pd.read_parquet(os.path.join(BASE, folder, f'{stem}.parquet'), columns=[
        'Run', 'Protein.Group', 'PG.MaxLFQ', 'Q.Value', 'PG.Q.Value'])
    d = d[(d['Q.Value'] < QVALUE) & (d['PG.Q.Value'] < QVALUE)]
    if d['Run'].nunique() != N_REPLICATES:
        raise ValueError(f'{label}: expected {N_REPLICATES} runs, '
                         f'found {d["Run"].nunique()}')

    pg = (d.groupby(['Protein.Group', 'Run'])['PG.MaxLFQ'].first()
          .groupby('Protein.Group').mean().rename('weight').reset_index())
    pg = pg[pg['weight'] > 0]
    pg['lead'] = pg['Protein.Group'].str.split(';').str[0]
    for acc in pg['lead'].unique():
        if acc not in cache:
            seq = sequences.get(acc)
            cache[acc] = properties_of(seq) if seq else (np.nan, np.nan, np.nan)
    for i, key in enumerate(('gravy', 'pi', 'length')):
        pg[key] = [cache[a][i] for a in pg['lead']]
    data[label] = (pg.dropna(subset=['gravy', 'pi', 'length'])
                   .set_index('Protein.Group'))
    print(f'{label:5s} {len(pg):>6,} protein groups, {len(data[label]):>6,} '
          f'with a FASTA sequence ({FORMAT[label]})')

# ---------------------------------------------------------------------------
# Weighted summaries
# ---------------------------------------------------------------------------
rows = []
for label in ORDER:
    frame = data[label]
    w = frame['weight'].to_numpy()
    for prop in PROPERTIES:
        v = frame[prop['key']].to_numpy()
        back = (lambda x: 10.0 ** x) if prop['log'] else (lambda x: x)
        row = {'condition': label, 'format': FORMAT[label],
               'property': prop['key'], 'n_protein_groups': len(v),
               'weighted_median': back(weighted_quantile(v, w, 0.5)),
               'weighted_q1': back(weighted_quantile(v, w, 0.25)),
               'weighted_q3': back(weighted_quantile(v, w, 0.75)),
               'unweighted_median': back(float(np.median(v)))}
        if prop['key'] == 'pi':
            row['signal_frac_acidic_pi_below_6'] = float(w[v < 6].sum() / w.sum())
            row['signal_frac_neutral_pi_6_to_8'] = float(
                w[(v >= 6) & (v <= 8)].sum() / w.sum())
            row['signal_frac_basic_pi_above_8'] = float(
                w[v > BASIC_PI].sum() / w.sum())
        rows.append(row)
summary = pd.DataFrame(rows)

for prop in PROPERTIES:
    sub = summary[summary['property'] == prop['key']]
    print(f'\nintensity-weighted {prop["key"]} (weighted median [Q1, Q3]):')
    for _, r in sub.iterrows():
        line = (f'  {r["condition"]:5s} {r["weighted_median"]:>8.3f} '
                f'[{r["weighted_q1"]:.3f}, {r["weighted_q3"]:.3f}]   '
                f'unweighted {r["unweighted_median"]:.3f}')
        if prop['key'] == 'pi':
            line += (f'   signal acidic/neutral/basic '
                     f'{r["signal_frac_acidic_pi_below_6"]:.1%} / '
                     f'{r["signal_frac_neutral_pi_6_to_8"]:.1%} / '
                     f'{r["signal_frac_basic_pi_above_8"]:.1%}')
        print(line)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
print('\nprotein groups seen by SAX, split on whether the phase saw them too:')
sax = data['SAX']
missed_rows = []
for label in ORDER:
    if label == 'SAX':
        continue
    seen = sax.index.intersection(data[label].index)
    missed = sax.index.difference(data[label].index)
    line = f'  {label:5s} missed {len(missed):>5,} of {len(sax):,}'
    entry = {'condition': label, 'n_missed': len(missed), 'n_seen': len(seen)}
    for prop in PROPERTIES:
        key = prop['key']
        back = (lambda x: 10.0 ** x) if prop['log'] else (lambda x: x)
        m_seen = back(weighted_quantile(sax.loc[seen, key].to_numpy(),
                                       sax.loc[seen, 'weight'].to_numpy(), 0.5))
        m_miss = back(weighted_quantile(sax.loc[missed, key].to_numpy(),
                                        sax.loc[missed, 'weight'].to_numpy(), 0.5))
        line += f'   {key} {m_seen:+.2f} -> {m_miss:+.2f}'
        entry[f'{key}_seen'] = m_seen
        entry[f'{key}_missed'] = m_miss
    entry['missed_signal_share_of_sax'] = float(
        sax.loc[missed, 'weight'].sum() / sax['weight'].sum())
    line += f'   carrying {entry["missed_signal_share_of_sax"]:.1%} of SAX signal'
    print(line)
    missed_rows.append(entry)

# ---------------------------------------------------------------------------
# Plot — three panels in one row, sized like the correlation-scatter row.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
curves = []
for ax, prop in zip(axes, PROPERTIES):
    key = prop['key']
    every = np.concatenate([data[c][key].to_numpy() for c in ORDER])
    lo, hi = np.percentile(every, [0.1, 99.9])
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, GRID)
    peak = 0.0
    for label in ORDER:
        v = data[label][key].to_numpy()
        w = data[label]['weight'].to_numpy()
        density = gaussian_kde(v, weights=w / w.sum())(grid)
        peak = max(peak, float(density.max()))
        ax.plot(grid, density, color=COLOR[label], linewidth=1.3,
                label=f'{label} ({legend_value(prop, v, w)})')
        curves.append(pd.DataFrame({
            'property': key, 'condition': label,
            'value': 10.0 ** grid if prop['log'] else grid,
            'intensity_weighted_density': density}))
    if prop['ref'] is not None:
        ax.axvline(prop['ref'], color='#999999', linestyle=':', linewidth=0.7,
                   zorder=1)
    if prop['log']:
        ticks = [t for t in LENGTH_TICKS if lo - pad <= np.log10(t) <= hi + pad]
        ax.xaxis.set_major_locator(FixedLocator([np.log10(t) for t in ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, peak * 1.55)
    ax.set_xlabel(prop['label'], fontsize=FONTSIZE)
    ax.set_ylabel('Intensity-weighted density', fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=6, handlelength=1.0,
              handletextpad=0.35, labelspacing=0.2, borderaxespad=0.15,
              title=prop['legend_title'], title_fontsize=6)
    ax.get_legend().get_title().set_color('#666666')

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')

pd.concat(curves, ignore_index=True).to_csv(
    os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)
summary.to_csv(os.path.join(OUTDIR, f'{STEM}_statistics.csv'), index=False)
pd.DataFrame(missed_rows).to_csv(
    os.path.join(OUTDIR, f'{STEM}_missed_vs_sax.csv'), index=False)

print(f'\nSaved {STEM} to {OUTDIR}')
