"""Figure 2, layout panel c — depth and quantification vs protein input (H032_E305)."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spec_analytics as core
import common_figure2 as cf

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
ROOT = os.path.join(INPUT, 'H032_E305')
OUTDIR = _cfg.output_dir(__file__)

METHODS = ['SPEC', 'PAC', 'ISD+']
DISPLAY = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}

INPUT_MAP = {'5ng': 5, '20ng': 20, '50ng': 50, '200ng': 200, '500ng': 500}
LEVELS = sorted(INPUT_MAP.values())
N_REPLICATES = 4
SCALE = 1e12
RIGHT_LABEL = 'Peptides at CV < 20 %'

# Method hues, shared with figure 2: SPEC coral, PAC sky, ISD+ pink.
COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}

# ---------------------------------------------------------------------------
# Per-run protein groups and summed raw precursor quantity.
# ---------------------------------------------------------------------------
rows, pep_rows = [], []
for method in METHODS:
    for sub, ng in INPUT_MAP.items():
        path = os.path.join(ROOT, method, sub, 'report.parquet')
        if not os.path.exists(path):
            path = os.path.join(ROOT, method, sub, f'{sub}.parquet')
        if not os.path.exists(path):
            raise FileNotFoundError(f'no combined report for {method} {sub}')
        d = pd.read_parquet(path, columns=[
            'Run', 'Protein.Group', 'Precursor.Quantity', 'Decoy', 'Q.Value',
            'PG.Q.Value'])
        d = d[(d['Decoy'] == 0) & (d['Q.Value'] <= 0.01)
              & (d['PG.Q.Value'] <= 0.01)]
        per_run_cond = d.groupby('Run').agg(
            protein_groups=('Protein.Group', 'nunique'),
            precursor_sum_raw=('Precursor.Quantity', 'sum'))
        if len(per_run_cond) != N_REPLICATES:
            raise ValueError(f'{method} {sub}: expected {N_REPLICATES} runs, '
                             f'found {len(per_run_cond)}')
        for rep, (run, r) in enumerate(sorted(per_run_cond.iterrows()), start=1):
            rows.append({'method': method, 'input_ng': ng, 'run': run,
                         'replicate': rep,
                         'protein_groups': int(r['protein_groups']),
                         'precursor_sum_raw': float(r['precursor_sum_raw'])})

        # Peptide level, from the definition shared with the supplement.
        summary = cf.summarise(cf.read_peptides(path))
        pep_rows.append({'method': method, 'input_ng': ng,
                         'peptides_per_run': int(np.mean(summary['per_run'])),
                         'peptides_cv20': summary['n_cv20'],
                         'median_cv_pct': summary['median_cv_pct']})

per_run = pd.DataFrame(rows)
pep = pd.DataFrame(pep_rows)

for metric in ['protein_groups', 'precursor_sum_raw']:
    table = per_run.pivot_table(index='input_ng', columns='method',
                               values=metric, aggfunc='mean').reindex(LEVELS)
    print(f'\nmean {metric} per run:')
    print((table[METHODS] / (SCALE if 'sum' in metric else 1)).round(2).to_string())

for metric in ['peptides_per_run', 'peptides_cv20', 'median_cv_pct']:
    print(f'\n{metric}:')
    print(pep.pivot_table(index='input_ng', columns='method', values=metric)
          .reindex(LEVELS)[METHODS].round(2).to_string())

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def replicates(method, column):
    return {a: per_run.loc[(per_run['method'] == method)
                           & (per_run['input_ng'] == a), column].to_numpy()
            for a in LEVELS}


def condition_value(method, column):
    """One value per input level. A CV has no per-replicate decomposition."""
    return {a: np.array([pep.loc[(pep['method'] == method)
                                 & (pep['input_ng'] == a), column].iloc[0]])
            for a in LEVELS}


series = [{'label': f'{m} — protein groups', 'method': m, 'axis': 'left',
           'color': COLOR[m], 'values': replicates(m, 'protein_groups')}
          for m in METHODS]
series += [{'label': f'{m} — peptides CV<20%', 'method': m, 'axis': 'right',
            'color': COLOR[m], 'values': condition_value(m, 'peptides_cv20')}
           for m in METHODS]

fig, ax_l = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
ax_r = ax_l.twinx()
heights, bar_points = cf.draw_grouped_dual(ax_l, ax_r, LEVELS, series)
cf.style_dual_axes(ax_l, ax_r, left_label='Protein groups',
                   right_label=RIGHT_LABEL,
                   left_max=per_run['protein_groups'].max(),
                   right_max=pep['peptides_cv20'].max())
ax_l.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax_r.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax_l.set_xlabel('Protein input amount [ng]', fontsize=cf.FONTSIZE)
cf.dual_legends(ax_l, cf.method_key([(DISPLAY[m], COLOR[m]) for m in METHODS]),
                cf.metric_key('Protein groups', 'Peptides CV < 20 %'))

fig.tight_layout()
cf.set_axes_size_inches(fig, [ax_l, ax_r], h_in=cf.AXES_H_IN)
cf.finish_points(ax_l, bar_points)
print(f'\nbar width {cf.bar_width_inches(fig, ax_l, len(METHODS), len(LEVELS)):.3f} in')
fig.savefig(os.path.join(OUTDIR, 'panel_c_pg_vs_input.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_c_pg_vs_input.png'), dpi=300,
            bbox_inches='tight')

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
cols = ['series', 'metric', 'method', 'input_ng', 'run', 'replicate', 'value']
label_of = {s['label']: s for s in series}
pts = pd.concat([
    per_run.assign(method=per_run['method'].map(DISPLAY), series='replicate point', metric='Protein groups',
                   value=per_run['protein_groups'])[cols],
    per_run.assign(method=per_run['method'].map(DISPLAY), series='replicate point',
                   metric='Summed precursor intensity [1e12]',
                   value=per_run['precursor_sum_raw'] / SCALE)[cols]],
    ignore_index=True)
bars = pd.DataFrame([
    {'series': 'bar height (mean)' if label_of[label]['axis'] == 'left'
     else 'line marker (condition)',
     'metric': ('Protein groups' if label_of[label]['axis'] == 'left'
                else 'Peptides at CV < 20 %'),
     'method': DISPLAY[label_of[label]['method']], 'input_ng': cat, 'run': '',
     'replicate': np.nan, 'value': mean}
    for label, cat, mean in heights])
# Not plotted, but what the text quotes, so it travels with the panel.
extra = pd.DataFrame([
    {'series': 'condition summary', 'metric': 'Median peptide CV [%]',
     'method': DISPLAY[r['method']], 'input_ng': r['input_ng'], 'run': '',
     'replicate': np.nan, 'value': round(r['median_cv_pct'], 2)}
    for _i, r in pep.iterrows()])
pd.concat([pts, bars[cols], extra[cols]], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'panel_c_pg_vs_input_sourcedata.csv'), index=False)

print(f'Saved the input panel to {OUTDIR}')
