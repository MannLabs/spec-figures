"""Supplementary figure 4d — detergent tolerance of the SAX phase (H032_E307)."""
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
BASE = os.path.join(INPUT, 'H032_E307', 'SAX')
OUTDIR = _cfg.output_dir_of('supplementary_figure4')
CONDITIONS = [
    ('10percent_SDS',  '10% SDS'),
    ('2percent_SDS',   '2% SDS'),
    ('0p5percent_SDS', '0.5% SDS'),
    ('4percent_SDC',   '4% SDC'),
    ('2percent_SDC',   '2% SDC'),
    ('no_detergent',   'no detergent'),
]
ORDER = [label for _f, label in CONDITIONS]
REFERENCE = 'no detergent'
QVALUE = 0.01
N_REPLICATES = 4
MIN_VALUES_FOR_CV = 3
SCALE = 1e12
RIGHT_LABEL = 'Peptides at CV < 20 %'
SAX_COLOR = core.PALETTE_SINGLE[0]
CLASS_GROUPS = [('no detergent', ['no detergent']),
                ('SDC', ['2% SDC', '4% SDC']),
                ('SDS', ['0.5% SDS', '2% SDS', '10% SDS'])]
PLOT_ORDER = [c for _lab, members in CLASS_GROUPS for c in members]
TICK = {'no detergent': 'none', '2% SDC': '2%', '4% SDC': '4%',
        '0.5% SDS': '0.5%', '2% SDS': '2%', '10% SDS': '10%'}
counts, prec_mat, cv_rows = {}, {}, []
for folder, label in CONDITIONS:
    path = os.path.join(BASE, folder, f'{folder}.parquet')
    raw = pd.read_parquet(path, columns=[
        'Run', 'Protein.Group', 'Precursor.Id', 'Precursor.Quantity',
        'PG.MaxLFQ', 'PG.Q.Value'])
    n_runs = raw['Run'].nunique()
    if n_runs != N_REPLICATES:
        raise ValueError(f'{label}: expected {N_REPLICATES} runs in {path}, '
                         f'found {n_runs}')
    raw = raw[raw['PG.Q.Value'] < QVALUE]
    counts[label] = raw.groupby('Run').agg(
        protein_groups=('Protein.Group', 'nunique'),
        precursor_sum_raw=('Precursor.Quantity', 'sum'))
    pos = raw[raw['Precursor.Quantity'] > 0]
    prec_mat[label] = pos.pivot_table(index='Precursor.Id', columns='Run',
                                      values='Precursor.Quantity', aggfunc='max')
    pg = raw.drop_duplicates(['Run', 'Protein.Group'])
    wide = pg.pivot(index='Protein.Group', columns='Run',
                    values='PG.MaxLFQ').replace(0, np.nan)
    wide = wide[wide.notna().sum(axis=1) >= MIN_VALUES_FOR_CV]
    cv = 100 * wide.std(axis=1, ddof=1) / wide.mean(axis=1)
    summary = cf.summarise(cf.read_peptides(path))
    cv_rows.append({'condition': label, 'n_protein_groups_for_cv': len(cv),
                    'median_pg_cv_pct': cv.median(),
                    'pg_cv20': int((cv < 100 * cf.CV_THRESHOLD).sum()),
                    'peptides_per_run': int(np.mean(summary['per_run'])),
                    'peptides_cv20': summary['n_cv20'],
                    'median_peptide_cv_pct': summary['median_cv_pct']})
    print(f'{label:13s} {n_runs} runs, PGs {counts[label]["protein_groups"].mean():,.0f}, '
          f'summed quantity {counts[label]["precursor_sum_raw"].mean() / SCALE:.2f}e12, '
          f'median PG CV {cv.median():.1f}%')
ctrl_median = prec_mat[REFERENCE].median(axis=1)
rows = []
for _folder, label in CONDITIONS:
    m = prec_mat[label]
    for rep, run in enumerate(sorted(m.columns), start=1):
        s = m[run].dropna()
        common = s.index.intersection(ctrl_median.index)
        rows.append({
            'condition': label, 'run': run, 'replicate': rep,
            'protein_groups': int(counts[label].loc[run, 'protein_groups']),
            'precursor_sum_raw': float(counts[label].loc[run, 'precursor_sum_raw']),
            'n_precursors_vs_control': len(common),
            'median_ratio_to_control_pct':
                100 * float((s[common] / ctrl_median[common]).median()),
        })
per_run = pd.DataFrame(rows)
for column in ('protein_groups', 'precursor_sum_raw'):
    ref = per_run.loc[per_run['condition'] == REFERENCE, column].mean()
    per_run[f'{column}_pct_of_control'] = 100 * per_run[column] / ref
summary = (per_run.groupby('condition')[
    ['protein_groups', 'precursor_sum_raw', 'protein_groups_pct_of_control',
     'precursor_sum_raw_pct_of_control', 'median_ratio_to_control_pct']]
    .agg(['mean', 'std']).reindex(ORDER))
print('\nabsolute values and % of the no-detergent control '
      f'(mean +/- SD over {N_REPLICATES} replicates):')
print(summary.round(2).to_string())
def replicates(column, scale=1.0):
    return {c: per_run.loc[per_run['condition'] == c, column].to_numpy() / scale
            for c in ORDER}
pep = pd.DataFrame(cv_rows).set_index('condition')
print('\npeptides per run, the CV < 20 % subset and the median peptide CV:')
print(pep[['peptides_per_run', 'peptides_cv20',
           'median_peptide_cv_pct']].reindex(ORDER).round(2).to_string())
series = [{'label': 'SAX SPEC', 'method': 'SAX SPEC', 'color': SAX_COLOR,
           'totals': replicates('protein_groups'),
           'cv20': {c: int(pep.loc[c, 'pg_cv20']) for c in ORDER}}]
fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
heights, bar_points = cs.draw_grouped_overlapping(
    ax, PLOT_ORDER, series, class_groups=CLASS_GROUPS)
ax.set_xticklabels([TICK[c] for c in PLOT_ORDER], fontsize=cf.FONTSIZE)
cs.style_axes(ax, xlabel='', ylabel='Protein groups',
              ymax=per_run['protein_groups'].max())
cf.dual_legends(ax, [], cs.metric_key(with_intensity=False,
                                      total_label='Protein groups'))
cs.finish(fig, ax, bar_points, 'suppl4_d_detergent_protein_groups',
          outdir=OUTDIR)
cv_by_cond = pd.DataFrame(cv_rows).set_index('condition')['median_pg_cv_pct']
METRIC = {'protein groups': 'Protein groups',
          'precursor intensity': 'Summed precursor intensity [% of control]',
          'peptides CV<20%': 'Peptides at CV < 20 %'}
cols = ['series', 'metric', 'condition', 'run', 'replicate', 'value',
        'pct_of_control', 'precursor_sum_raw_1e12', 'median_pg_cv_pct']
pts = pd.concat([
    per_run.assign(series='replicate point', metric=metric, value=per_run[col],
                   pct_of_control=per_run.get(f'{col}_pct_of_control',
                                              per_run[col]),
                   precursor_sum_raw_1e12=per_run['precursor_sum_raw'] / SCALE,
                   median_pg_cv_pct=per_run['condition'].map(cv_by_cond))[cols]
    for col, metric in (
        ('protein_groups', METRIC['protein groups']),
        ('precursor_sum_raw_pct_of_control', METRIC['precursor intensity']))],
    ignore_index=True)
bars = pd.DataFrame(
    [{'series': 'light bar (mean)', 'metric': METRIC['protein groups'],
      'condition': cat, 'run': '', 'replicate': np.nan, 'value': mean,
      'pct_of_control': np.nan, 'precursor_sum_raw_1e12': np.nan,
      'median_pg_cv_pct': cv_by_cond[cat]}
     for _label, cat, mean, _dark in heights]
    + [{'series': 'dark bar (condition)',
        'metric': 'Protein groups at CV < 20 %', 'condition': cat, 'run': '',
        'replicate': np.nan, 'value': dark, 'pct_of_control': np.nan,
        'precursor_sum_raw_1e12': np.nan, 'median_pg_cv_pct': cv_by_cond[cat]}
       for _label, cat, _mean, dark in heights])
extra = pd.DataFrame([
    {'series': 'condition summary', 'metric': m, 'condition': c, 'run': '',
     'replicate': np.nan, 'value': v, 'pct_of_control': np.nan,
     'precursor_sum_raw_1e12': np.nan, 'median_pg_cv_pct': cv_by_cond[c]}
    for c in ORDER
    for m, v in (('Peptides per run', pep.loc[c, 'peptides_per_run']),
                 ('Median peptide CV [%]',
                  round(pep.loc[c, 'median_peptide_cv_pct'], 2)))])
pd.concat([pts, bars[cols], extra[cols]], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'suppl4_d_detergent_protein_groups_sourcedata.csv'),
    index=False)
print(f'Saved the detergent panel to {OUTDIR}')
