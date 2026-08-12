"""Supplementary figure 4a + 4b — summed precursor intensity per run."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyarrow.parquet as pq

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
OUTDIR = _cfg.output_dir(__file__)

QVALUE = 0.01
FONTSIZE = 8

ROW_CONDITION = {'A': 'Single-shot SPEC', 'B': 'Single-shot ISD+',
                 'C': 'Bulk ISD+'}
ORDER = ['Single-shot ISD+', 'Bulk ISD+', 'Single-shot SPEC']
XTICK = {'Single-shot ISD+': 'Single-shot\nISD+', 'Bulk ISD+': 'Bulk\nISD+',
         'Single-shot SPEC': 'Single-shot\nSPEC'}

ROW_ORGAN = {'A': 'Liver', 'B': 'Brain', 'C': 'Heart', 'D': 'Kidney',
             'E': 'Testis', 'F': 'Lung'}
METHODS = ['PAC', 'SPEC']
DODGE = {'PAC': -0.16, 'SPEC': 0.16}


def lighten(color, factor=0.45):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * factor)


# Hues locked to figure 4: SPEC coral, PAC sky, ISD+ pink (lightened for the
# single-shot variant so the two ISD+ conditions read as one method).
COLOR_A = {'Single-shot SPEC': core.PALETTE_SINGLE[0],
           'Bulk ISD+': core.PALETTE_SINGLE[5],
           'Single-shot ISD+': lighten(core.PALETTE_SINGLE[5])}
COLOR_B = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2]}

YLABEL = r'$\log_2$ summed precursor intensity'

YLIM = (33.0, 41.2)
YTICKS = [34, 36, 38, 40]


def read_filtered(path, extra=()):
    """Load a report with figure 4's q-value filters, needed columns only."""
    available = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = [c for c in (('Run', 'Precursor.Id', 'Precursor.Quantity',
                           'Decoy', 'Q.Value', 'PG.Q.Value') + tuple(extra))
              if c in available]
    raw = pd.read_parquet(path, columns=wanted)
    mask = (raw['Q.Value'] < QVALUE) & (raw['PG.Q.Value'] < QVALUE)
    if 'Decoy' in raw:
        mask &= raw['Decoy'] == 0
    else:
        print(f'note: no Decoy column in {os.path.basename(path)}; '
              'q-value filters only')
    return raw[mask].copy()


def run_totals(raw, keys):
    """Total precursor signal per run, over unique (run, precursor) pairs.

    `Precursor.Quantity` is the raw quantity, deliberately not
    `Precursor.Normalised`: the panel exists to show how much signal each
    preparation recovered, and the normalised column has been rescaled across runs,
    which is exactly the between-condition difference the panel should display.
    """
    one = raw.drop_duplicates(['Run', 'Precursor.Id'])
    quant = one['Precursor.Quantity'].astype(float).replace(0, np.nan)
    out = (one.assign(q=quant).groupby(keys, observed=True)
           .agg(total_precursor_quantity=('q', 'sum'),
                n_precursors=('q', 'count')).reset_index()
           .rename(columns={'Run': 'run'}))
    out['log2_total'] = np.log2(out['total_precursor_quantity'])
    return out


def finish(fig, ax, name):
    ax.set_ylabel(YLABEL, fontsize=FONTSIZE)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.tick_params(labelsize=FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{name}.png'), dpi=300, bbox_inches='tight')


# ---------------------------------------------------------------------------
# a — the three preparations of figure 4b
# ---------------------------------------------------------------------------
print('panel a — H032_E127')
FIG4_INPUT = _cfg.cross_input('figure4')
FFPE_REPORTS = {
    'Single-shot SPEC': os.path.join(FIG4_INPUT, 'H032_E127',
                                     'single-shot_SPEC', 'report.parquet'),
    'Single-shot ISD+': os.path.join(FIG4_INPUT, 'H032_E127',
                                     'single-shot_ISD+', 'report.parquet'),
    'Bulk ISD+': os.path.join(FIG4_INPUT, 'H032_E127',
                              'bulk_ISD+', 'report.parquet')}
ORGAN_REPORTS = {
    'SPEC': os.path.join(FIG4_INPUT, 'H032_E170', 'SPEC', 'report.parquet'),
    'PAC': os.path.join(FIG4_INPUT, 'H032_E170', 'PAC', 'report.parquet')}

raw = pd.concat([read_filtered(path).assign(condition=cond)
                 for cond, path in FFPE_REPORTS.items()], ignore_index=True)
raw['tag'] = raw['Run'].str[-2:]
raw['replicate'] = raw['tag'].str[1].astype(int)
tot_a = run_totals(raw, ['Run', 'condition', 'replicate'])

print(tot_a.groupby('condition')['log2_total']
      .agg(['count', 'mean', 'std']).reindex(ORDER).round(2).to_string())

PANEL_SIZE_A = (2.2, 3.2)
PANEL_SIZE_B = (4.0, 3.2)
MEAN_MARKER_PT = 10

rng = np.random.default_rng(0)
span = YLIM[1] - YLIM[0]
fig, ax = plt.subplots(figsize=PANEL_SIZE_A)
for i, cond in enumerate(ORDER):
    vals = tot_a.loc[tot_a['condition'] == cond, 'log2_total'].to_numpy()
    ax.plot(i, vals.mean(), marker='_', markersize=MEAN_MARKER_PT,
            markeredgecolor=COLOR_A[cond], markeredgewidth=2.2, zorder=3)
    ax.scatter(np.full(len(vals), i) + rng.uniform(-0.08, 0.08, len(vals)), vals,
               s=18, color=COLOR_A[cond], alpha=0.9, edgecolor='black',
               linewidth=0.4, zorder=5)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([XTICK[c] for c in ORDER], fontsize=FONTSIZE, rotation=45,
                   ha='right')
ax.set_xlim(-0.6, len(ORDER) - 0.4)
finish(fig, ax, 'panel_a_total_intensity_ffpe')

tot_a.assign(series='replicate point')[
    ['run', 'condition', 'replicate', 'n_precursors',
     'total_precursor_quantity', 'log2_total', 'series']].to_csv(
    os.path.join(OUTDIR, 'panel_a_total_intensity_ffpe_sourcedata.csv'),
    index=False)

# ---------------------------------------------------------------------------
# b — the six organs of figure 4f
# ---------------------------------------------------------------------------
print('\npanel b — H032_E170')
raw = pd.concat([read_filtered(path).assign(method=method)
                 for method, path in ORGAN_REPORTS.items()], ignore_index=True)
raw['tag'] = raw['Run'].str.extract(r'_([A-G]\d{1,2})$')[0]
raw['organ'] = raw['tag'].str[0].map(ROW_ORGAN)
if raw['organ'].isna().any():
    raise ValueError('runs whose plate row maps to no organ')
tot_b = run_totals(raw, ['Run', 'organ', 'method'])

ORGAN_ORDER = ['Testis', 'Kidney', 'Brain', 'Lung', 'Liver', 'Heart']
missing = set(tot_b['organ']) - set(ORGAN_ORDER)
if missing:
    raise ValueError(f'organs not in the figure 4f order: {missing}')
print(tot_b.pivot_table(index='organ', columns='method', values='log2_total',
                        aggfunc='mean').reindex(ORGAN_ORDER).round(2).to_string())

xpos = {organ: i for i, organ in enumerate(ORGAN_ORDER)}
fig, ax = plt.subplots(figsize=PANEL_SIZE_B)
for method in METHODS:
    sub = tot_b[tot_b['method'] == method]
    x = (sub['organ'].map(xpos) + DODGE[method]
         + rng.uniform(-0.05, 0.05, size=len(sub)))
    ax.scatter(x, sub['log2_total'], s=18, color=COLOR_B[method], alpha=0.85,
               edgecolor='black', linewidth=0.4, label=method, zorder=4)
    means = sub.groupby('organ')['log2_total'].mean()
    for organ, value in means.items():
        ax.plot(xpos[organ] + DODGE[method], value, marker='_',
                markersize=MEAN_MARKER_PT, markeredgecolor=COLOR_B[method],
                markeredgewidth=2.0, zorder=3)
ax.set_xticks(range(len(ORGAN_ORDER)))
ax.set_xticklabels(ORGAN_ORDER, fontsize=FONTSIZE, rotation=45, ha='right')
ax.set_xlim(-0.5, len(ORGAN_ORDER) - 0.5)
ax.legend(loc='lower right', frameon=False, fontsize=FONTSIZE, handlelength=0.9,
          handletextpad=0.4, borderpad=0.2, ncol=2, columnspacing=1.0)
finish(fig, ax, 'panel_b_total_intensity_organs')

tot_b.assign(series='replicate point')[
    ['run', 'organ', 'method', 'n_precursors', 'total_precursor_quantity',
     'log2_total', 'series']].to_csv(
    os.path.join(OUTDIR, 'panel_b_total_intensity_organs_sourcedata.csv'),
    index=False)

delta = (tot_b.groupby(['organ', 'method'])['log2_total'].mean().unstack()
         .reindex(ORGAN_ORDER))
delta['SPEC - PAC'] = delta['SPEC'] - delta['PAC']
print('\nlog2 signal difference, SPEC vs PAC:')
print(delta.round(2).to_string())
print(f'\nSaved panels a and b to {OUTDIR}')
