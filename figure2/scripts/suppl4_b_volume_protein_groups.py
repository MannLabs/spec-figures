"""Supplementary figure 4b — depth and quantification vs sample volume (H032_E306)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spec_analytics as core
import common_figure2 as cf
import sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'supplementary_figure3', 'scripts')))
import common_suppl3 as cs
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
ROOT = os.path.join(INPUT, 'H032_E306')
OUTDIR = _cfg.output_dir_of('supplementary_figure4')
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
        pg_cv = cf.protein_group_cv(path, runs=keep_runs)
        summary = cf.summarise(cf.read_peptides(path, runs=keep_runs))
        pep_rows.append({'method': method, 'volume_uL': vol,
                         'pg_cv20': cf.n_cv20(pg_cv),
                         'median_pg_cv_pct': float(100 * pg_cv.median()),
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
for metric in ['pg_cv20', 'median_pg_cv_pct', 'peptides_per_run',
               'peptides_cv20', 'median_cv_pct']:
    print(f'\n{metric}:')
    print(pep.pivot_table(index='volume_uL', columns='method', values=metric)
          .reindex(LEVELS)[METHODS].round(2).to_string())
print('\ndropped run per condition:')
print(pep.set_index(['method', 'volume_uL'])['dropped_run'].to_string())
def replicates(method, column):
    return {v: per_run.loc[(per_run['method'] == method)
                           & (per_run['volume_uL'] == v), column].to_numpy()
            for v in LEVELS}
def cv20(method):
    return {v: int(pep.loc[(pep['method'] == method)
                           & (pep['volume_uL'] == v), 'pg_cv20'].iloc[0])
            for v in LEVELS}
series = [{'label': DISPLAY[m], 'method': m, 'color': COLOR[m],
           'totals': replicates(m, 'protein_groups'), 'cv20': cv20(m)}
          for m in METHODS]
fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
heights, bar_points = cs.draw_grouped_overlapping(ax, LEVELS, series)
cs.style_axes(ax, xlabel='Sample volume [µL]', ylabel='Protein groups',
              ymax=per_run['protein_groups'].max())
cf.dual_legends(ax, cf.method_key([(DISPLAY[m], COLOR[m]) for m in METHODS]),
                cs.metric_key(with_intensity=False,
                              total_label='Protein groups'))
cs.finish(fig, ax, bar_points, 'suppl4_b_volume_protein_groups', outdir=OUTDIR)
cols = ['series', 'metric', 'method', 'volume_uL', 'run', 'replicate', 'value']
pts = pd.concat([
    per_run.assign(method=per_run['method'].map(DISPLAY),
                   series='replicate point', metric='Protein groups',
                   value=per_run['protein_groups'])[cols],
    per_run.assign(method=per_run['method'].map(DISPLAY),
                   series='replicate point',
                   metric='Summed precursor intensity [1e12]',
                   value=per_run['precursor_sum_raw'] / SCALE)[cols]],
    ignore_index=True)
bars = pd.DataFrame(
    [{'series': 'light bar (mean)', 'metric': 'Protein groups',
      'method': label, 'volume_uL': cat, 'run': '', 'replicate': np.nan,
      'value': mean} for label, cat, mean, _dark in heights]
    + [{'series': 'dark bar (condition)',
        'metric': 'Protein groups at CV < 20 %', 'method': label,
        'volume_uL': cat, 'run': '', 'replicate': np.nan, 'value': dark}
       for label, cat, _mean, dark in heights])
extra = pd.DataFrame([
    {'series': 'condition summary', 'metric': metric,
     'method': DISPLAY[r['method']], 'volume_uL': r['volume_uL'], 'run': '',
     'replicate': np.nan, 'value': round(r[column], 2)}
    for _i, r in pep.iterrows()
    for metric, column in (('Median protein-group CV [%]', 'median_pg_cv_pct'),
                           ('Median peptide CV [%]', 'median_cv_pct'),
                           ('Peptides per run', 'peptides_per_run'),
                           ('Peptides at CV < 20 %', 'peptides_cv20'))])
pd.concat([pts, bars[cols], extra[cols]], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'suppl4_b_volume_protein_groups_sourcedata.csv'),
    index=False)
print(f'Saved the volume panel to {OUTDIR}')
