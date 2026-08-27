"""Supplementary figure 4c — depth and quantification vs protein input (H032_E305)."""
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
ROOT = os.path.join(INPUT, 'H032_E305')
OUTDIR = _cfg.output_dir_of('supplementary_figure4')
METHODS = ['SPEC', 'PAC', 'ISD+']
DISPLAY = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}
INPUT_MAP = {'5ng': 5, '20ng': 20, '50ng': 50, '200ng': 200, '500ng': 500}
LEVELS = sorted(INPUT_MAP.values())
N_REPLICATES = 4
SCALE = 1e12
RIGHT_LABEL = 'Peptides at CV < 20 %'
COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}
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
        pg_cv = cf.protein_group_cv(path)
        summary = cf.summarise(cf.read_peptides(path))
        pep_rows.append({'method': method, 'input_ng': ng,
                         'pg_cv20': cf.n_cv20(pg_cv),
                         'median_pg_cv_pct': float(100 * pg_cv.median()),
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
def replicates(method, column):
    return {ng: per_run.loc[(per_run['method'] == method)
                            & (per_run['input_ng'] == ng), column].to_numpy()
            for ng in LEVELS}
def cv20(method):
    return {ng: int(pep.loc[(pep['method'] == method)
                            & (pep['input_ng'] == ng), 'pg_cv20'].iloc[0])
            for ng in LEVELS}
series = [{'label': DISPLAY[m], 'method': m, 'color': COLOR[m],
           'totals': replicates(m, 'protein_groups'), 'cv20': cv20(m)}
          for m in METHODS]
fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
heights, bar_points = cs.draw_grouped_overlapping(ax, LEVELS, series)
cs.style_axes(ax, xlabel='Protein input amount [ng]',
              ylabel='Protein groups', ymax=per_run['protein_groups'].max())
cf.dual_legends(ax, cf.method_key([(DISPLAY[m], COLOR[m]) for m in METHODS]),
                cs.metric_key(with_intensity=False,
                              total_label='Protein groups'))
cs.finish(fig, ax, bar_points, 'suppl4_c_input_protein_groups', outdir=OUTDIR)
cols = ['series', 'metric', 'method', 'input_ng', 'run', 'replicate', 'value']
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
      'method': label, 'input_ng': cat, 'run': '', 'replicate': np.nan,
      'value': mean} for label, cat, mean, _dark in heights]
    + [{'series': 'dark bar (condition)',
        'metric': 'Protein groups at CV < 20 %', 'method': label,
        'input_ng': cat, 'run': '', 'replicate': np.nan, 'value': dark}
       for label, cat, _mean, dark in heights])
extra = pd.DataFrame([
    {'series': 'condition summary', 'metric': metric,
     'method': DISPLAY[r['method']], 'input_ng': r['input_ng'], 'run': '',
     'replicate': np.nan, 'value': round(r[column], 2)}
    for _i, r in pep.iterrows()
    for metric, column in (('Median protein-group CV [%]', 'median_pg_cv_pct'),
                           ('Median peptide CV [%]', 'median_cv_pct'),
                           ('Peptides per run', 'peptides_per_run'),
                           ('Peptides at CV < 20 %', 'peptides_cv20'))])
pd.concat([pts, bars[cols], extra[cols]], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'suppl4_c_input_protein_groups_sourcedata.csv'),
    index=False)
print(f'Saved the input panel to {OUTDIR}')
