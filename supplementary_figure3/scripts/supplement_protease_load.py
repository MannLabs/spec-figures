"""Supplementary figure 3a-c — protease load against sample input (H032_E295)."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spec_analytics as core
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.cross_input('supplementary_figure1', 'H032_E295')
OUTDIR = _cfg.output_dir(__file__)
QVALUE = 0.01
INPUTS = [
    ('B', '5ng', '5 ng'),
    ('C', '200ng', '200 ng'),
    ('D', '5ug', '5 µg'),
]
INPUT_ORDER = [label for _row, _folder, label in INPUTS]
ROW_TO_INPUT = {row: label for row, _folder, label in INPUTS}
PROTEASE = [(12.5, '12.5ng', range(1, 5)),
            (50.0, '50ng', range(5, 9)),
            (200.0, '200ng', range(9, 13))]
PROTEASE_ORDER = [load for load, _suffix, _cols in PROTEASE]
COL_TO_PROTEASE = {c: load for load, _suffix, cols in PROTEASE for c in cols}
REPORTS = {(label, load): os.path.join(INPUT, folder, f'protease_{suffix}',
                                       'report.parquet')
           for _row, folder, label in INPUTS
           for load, suffix, _cols in PROTEASE}
FONTSIZE = 10
LEGEND_FONTSIZE = 9.5
STEM = 'supplement_protease_load'
PANELS = [
    ('protein_groups', 'Protein groups', '{:,.0f}'),
    ('mc0_by_intensity', 'Fully cleaved signal [%]', '{:.1f}'),
]
PANEL_W_IN = 4.20
def ramp(color, n):
    """Light-to-dark sequential shades of one hue, for an ordered variable."""
    base = np.array(mcolors.to_rgb(color))
    factors = np.linspace(0.62, 0.0, n)
    return [tuple(1 - (1 - base) * (1 - f)) for f in factors]
COLOR = dict(zip(INPUT_ORDER, ramp(core.PALETTE_SINGLE[0], len(INPUT_ORDER))))
COLUMNS = ['Run', 'Protein.Group', 'Precursor.Id', 'Stripped.Sequence',
           'Precursor.Quantity', 'Q.Value', 'PG.Q.Value']
mc_memo = {}
rows = []
for (label, load), path in REPORTS.items():
    raw = pd.read_parquet(path, columns=COLUMNS)
    raw = raw[(raw['Q.Value'] < QVALUE) & (raw['PG.Q.Value'] < QVALUE)
              & (raw['Precursor.Quantity'] > 0)].copy()
    raw['well'] = raw['Run'].str.extract(r'_([A-E]\d{1,2})$')[0]
    if raw['well'].isna().any():
        raise ValueError(f'{path}: runs whose plate well could not be parsed')
    raw['row'] = raw['well'].str[0]
    raw['col'] = raw['well'].str[1:].astype(int)
    cells = {(ROW_TO_INPUT.get(r), COL_TO_PROTEASE.get(c))
             for r, c in zip(raw['row'], raw['col'])}
    if cells != {(label, load)}:
        raise ValueError(f'{path}: folder says {(label, load)} but the wells say '
                         f'{sorted(cells)}')
    for seq in raw['Stripped.Sequence'].unique():
        if seq not in mc_memo:
            mc_memo[seq] = core.count_missed_cleavages(seq, protease='trypsin')
    raw['is_mc0'] = raw['Stripped.Sequence'].map(mc_memo) == 0
    for well, g in raw.groupby('well', sort=True):
        rows.append({
            'row': well[0], 'col': int(well[1:]), 'well': well,
            'protein_input': label, 'protease_ng': load,
            'protein_groups': g['Protein.Group'].nunique(),
            'precursors': g['Precursor.Id'].nunique(),
            'signal': g['Precursor.Quantity'].sum(),
            'mc0_by_count': 100 * g['is_mc0'].mean(),
            'mc0_by_intensity': (100 * g.loc[g['is_mc0'], 'Precursor.Quantity'].sum()
                                 / g['Precursor.Quantity'].sum()),
        })
per_run = (pd.DataFrame(rows)
           .sort_values(['row', 'col'], ignore_index=True))
grid = per_run
counts = per_run.groupby(['protein_input', 'protease_ng']).size()
if not (counts == 4).all():
    raise ValueError(f'expected 4 replicates per cell: '
                     f'{counts[counts != 4].to_dict()}')
print(f'{len(per_run)} runs across {len(REPORTS)} searches, '
      f'{len(counts)} cells of the input x protease grid')
for metric, label, fmt in PANELS:
    print(f'\n{label} (mean of 4 replicates):')
    table = grid.pivot_table(index='protein_input', columns='protease_ng',
                             values=metric, aggfunc='mean').reindex(INPUT_ORDER)
    print(table.round(1).to_string())
    gain = 100 * (table[200.0] / table[50.0] - 1)
    print('  50 -> 200 ng protease: ' +
          ', '.join(f'{i} {g:+.1f}%' for i, g in gain.items()))
x = np.arange(len(PROTEASE_ORDER))
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, len(PANELS),
                         figsize=(PANEL_W_IN * len(PANELS), 4))
rows_out = []
for ax, (metric, ylabel, _fmt) in zip(axes, PANELS):
    for k, label in enumerate(INPUT_ORDER):
        sub = grid[grid['protein_input'] == label]
        offset = (k - (len(INPUT_ORDER) - 1) / 2) * 0.17
        for i, load in enumerate(PROTEASE_ORDER):
            vals = sub.loc[sub['protease_ng'] == load, metric].to_numpy()
            jitter = rng.uniform(-0.035, 0.035, size=len(vals))
            ax.scatter(np.full(len(vals), x[i] + offset) + jitter, vals, s=34,
                       color=COLOR[label], edgecolor='black', linewidth=0.3,
                       alpha=0.9, zorder=4,
                       label=label if i == 0 else None)
            rows_out.extend({'panel': metric, 'protein_input': label,
                             'protease_ng': load, 'value': float(v)}
                            for v in vals)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:g}' for v in PROTEASE_ORDER], fontsize=FONTSIZE)
    ax.set_xlim(-0.5, len(PROTEASE_ORDER) - 0.5)
    ax.set_ylim(0, None)
    ax.set_xlabel('Total protease [ng]', fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{int(v):,}' if v >= 1000 else f'{v:g}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].legend(loc='lower right', frameon=False, fontsize=LEGEND_FONTSIZE,
               handlelength=0.8, handletextpad=0.35, labelspacing=0.25,
               borderaxespad=0.2, title='protein input',
               title_fontsize=LEGEND_FONTSIZE)
axes[0].get_legend().get_title().set_color('#666666')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')
pd.DataFrame(rows_out).to_csv(
    os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)
per_run.to_csv(os.path.join(OUTDIR, f'{STEM}_per_run.csv'), index=False)
print(f'\nSaved {STEM} to {OUTDIR}')
