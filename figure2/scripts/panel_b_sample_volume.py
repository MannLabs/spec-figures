"""Figure 2, layout panel b — depth and quantification vs sample volume (H032_E306)."""

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
ROOT = os.path.join(INPUT, 'H032_E306')
OUTDIR = _cfg.output_dir(__file__)

# SPEC first, as in every method comparison of this figure. The in-solution arm is
# **ISD+**: the methods specify an SDB-RPS cleanup for this experiment.
METHODS = ['SPEC', 'PAC', 'ISD+']
DISPLAY = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}

VOLUMES = {'5uL': 5, '10uL': 10, '40uL': 40, '100uL': 100, '200uL': 200}
LEVELS = list(VOLUMES.values())
QVALUE = 0.01
N_ACQUIRED = 4
EXCLUDE_REPLICATE = 3
N_REPLICATES = N_ACQUIRED - 1
SCALE = 1e12
RIGHT_LABEL = 'Peptides at CV < 20 %'

COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}

# ---------------------------------------------------------------------------
# Per-run protein groups and summed quantity; per-condition peptide CVs.
# ---------------------------------------------------------------------------
rows, pep_rows = [], []
for method in METHODS:
    for vol_name, vol in VOLUMES.items():
        path = os.path.join(ROOT, method, vol_name, f'{vol_name}.parquet')
        raw = pd.read_parquet(path, columns=[
            'Run', 'Protein.Group', 'Precursor.Quantity', 'PG.Q.Value'])
        runs = sorted(raw['Run'].unique())
        if len(runs) != N_ACQUIRED:
            raise ValueError(f'{method} {vol_name}: expected {N_ACQUIRED} runs, '
                             f'found {len(runs)}')
        dropped = runs[EXCLUDE_REPLICATE - 1]
        keep_runs = [r for r in runs if r != dropped]

        keep = raw[(raw['PG.Q.Value'] < QVALUE) & raw['Run'].isin(keep_runs)]
        per_run_cond = keep.groupby('Run').agg(
            protein_groups=('Protein.Group', 'nunique'),
            precursor_sum_raw=('Precursor.Quantity', 'sum'))
        for rep, (run, r) in enumerate(sorted(per_run_cond.iterrows()), start=1):
            rows.append({'method': method, 'volume_uL': vol, 'run': run,
                         'replicate': rep,
                         'protein_groups': int(r['protein_groups']),
                         'precursor_sum_raw': float(r['precursor_sum_raw'])})

        # Peptide level, from the same definition the supplement plots.
        summary = cf.summarise(cf.read_peptides(path, runs=keep_runs))
        pep_rows.append({'method': method, 'volume_uL': vol,
                         'peptides_per_run': int(np.mean(summary['per_run'])),
                         'peptides_cv20': summary['n_cv20'],
                         'median_cv_pct': summary['median_cv_pct'],
                         'dropped_run': dropped})

per_run = pd.DataFrame(rows)
pep = pd.DataFrame(pep_rows)

for metric in ['protein_groups', 'precursor_sum_raw']:
    table = per_run.pivot_table(index='volume_uL', columns='method',
                                values=metric, aggfunc='mean').reindex(LEVELS)
    print(f'\nmean {metric} per run (n = {N_REPLICATES}):')
    print((table[METHODS] / (SCALE if 'sum' in metric else 1)).round(2).to_string())

for metric in ['peptides_per_run', 'peptides_cv20', 'median_cv_pct']:
    print(f'\n{metric}:')
    print(pep.pivot_table(index='volume_uL', columns='method', values=metric)
          .reindex(LEVELS)[METHODS].round(2).to_string())

print('\ndropped run per condition:')
print(pep.set_index(['method', 'volume_uL'])['dropped_run'].to_string())


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def replicates(method, column):
    return {v: per_run.loc[(per_run['method'] == method)
                           & (per_run['volume_uL'] == v), column].to_numpy()
            for v in LEVELS}


def condition_value(method, column):
    """One value per volume. No per-replicate decomposition exists for a CV."""
    return {v: np.array([pep.loc[(pep['method'] == method)
                                 & (pep['volume_uL'] == v), column].iloc[0]])
            for v in LEVELS}


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
ax_l.set_xlabel('Sample volume [µL]', fontsize=cf.FONTSIZE)
cf.dual_legends(ax_l, cf.method_key([(DISPLAY[m], COLOR[m]) for m in METHODS]),
                cf.metric_key('Protein groups', 'Peptides CV < 20 %'))

fig.tight_layout()
cf.set_axes_size_inches(fig, [ax_l, ax_r], h_in=cf.AXES_H_IN)
cf.finish_points(ax_l, bar_points)
print(f'\nbar width {cf.bar_width_inches(fig, ax_l, len(METHODS), len(LEVELS)):.3f} in')
fig.savefig(os.path.join(OUTDIR, 'panel_b_sample_volume.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_b_sample_volume.png'), dpi=300,
            bbox_inches='tight')

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
cols = ['series', 'metric', 'method', 'volume_uL', 'run', 'replicate', 'value']
label_of = {s['label']: s for s in series}
pts = pd.concat([
    per_run.assign(method=per_run['method'].map(DISPLAY),
                   series='replicate point', metric='Protein groups',
                   value=per_run['protein_groups'])[cols],
    per_run.assign(method=per_run['method'].map(DISPLAY),
                   series='replicate point',
                   metric='Summed precursor intensity [1e12]',
                   value=per_run['precursor_sum_raw'] / SCALE)[cols]],
    ignore_index=True)
bars = pd.DataFrame([
    {'series': 'bar height (mean)' if label_of[label]['axis'] == 'left'
     else 'line marker (condition)',
     'metric': ('Protein groups' if label_of[label]['axis'] == 'left'
                else 'Peptides at CV < 20 %'),
     'method': DISPLAY[label_of[label]['method']], 'volume_uL': cat, 'run': '',
     'replicate': np.nan, 'value': mean}
    for label, cat, mean in heights])
# The median CV is not plotted but is what the text quotes, so it travels with the
# panel rather than only in the supplement.
extra = pd.DataFrame([
    {'series': 'condition summary', 'metric': 'Median peptide CV [%]',
     'method': DISPLAY[r['method']], 'volume_uL': r['volume_uL'], 'run': '',
     'replicate': np.nan, 'value': round(r['median_cv_pct'], 2)}
    for _i, r in pep.iterrows()])
pd.concat([pts, bars[cols], extra[cols]], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'panel_b_sample_volume_sourcedata.csv'), index=False)

print(f'Saved the volume panel to {OUTDIR}')
