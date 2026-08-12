"""Figure 2, layout panel d — detergent tolerance of the SAX phase (H032_E307)."""

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
BASE = os.path.join(INPUT, 'H032_E307', 'SAX')
OUTDIR = _cfg.output_dir(__file__)

# (folder, display label) in plotting order.
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

SAX_COLOR = core.PALETTE_SINGLE[0]   # coral

# ---------------------------------------------------------------------------
# Per-condition PG counts, summed quantity, raw precursor matrix, median PG CV.
# ---------------------------------------------------------------------------
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

    # CV on linear PG intensities, SD/mean with ddof=1.
    pg = raw.drop_duplicates(['Run', 'Protein.Group'])
    wide = pg.pivot(index='Protein.Group', columns='Run',
                    values='PG.MaxLFQ').replace(0, np.nan)
    wide = wide[wide.notna().sum(axis=1) >= MIN_VALUES_FOR_CV]
    cv = 100 * wide.std(axis=1, ddof=1) / wide.mean(axis=1)
    # Peptide level, from the definition shared with the supplement.
    summary = cf.summarise(cf.read_peptides(path))
    cv_rows.append({'condition': label, 'n_protein_groups_for_cv': len(cv),
                    'median_pg_cv_pct': cv.median(),
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

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def replicates(column, scale=1.0):
    return {c: per_run.loc[per_run['condition'] == c, column].to_numpy() / scale
            for c in ORDER}


pep = pd.DataFrame(cv_rows).set_index('condition')
print('\npeptides per run, the CV < 20 % subset and the median peptide CV:')
print(pep[['peptides_per_run', 'peptides_cv20',
           'median_peptide_cv_pct']].reindex(ORDER).round(2).to_string())

series = [
    {'label': 'protein groups', 'method': 'SAX SPEC', 'axis': 'left',
     'color': SAX_COLOR, 'values': replicates('protein_groups')},
    # A CV is defined at the condition level, so this series carries one value per
    # detergent and no replicate scatter.
    {'label': 'peptides CV<20%', 'method': 'SAX SPEC', 'axis': 'right',
     'color': SAX_COLOR,
     'values': {c: np.array([pep.loc[c, 'peptides_cv20']]) for c in ORDER}},
]

fig, ax_l = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
ax_r = ax_l.twinx()
heights, bar_points = cf.draw_grouped_dual(ax_l, ax_r, ORDER, series)
cf.style_dual_axes(ax_l, ax_r, left_label='Protein groups',
                   right_label=RIGHT_LABEL,
                   left_max=per_run['protein_groups'].max(),
                   right_max=pep['peptides_cv20'].max())
ax_l.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax_r.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
plt.setp(ax_l.get_xticklabels(), rotation=45, ha='right')
cf.dual_legends(ax_l, [], cf.metric_key('Protein groups', 'Peptides CV < 20 %'))

fig.tight_layout()
cf.set_axes_size_inches(fig, [ax_l, ax_r], h_in=cf.AXES_H_IN)
cf.finish_points(ax_l, bar_points)
print(f'\nbar width {cf.bar_width_inches(fig, ax_l, 1, len(ORDER)):.3f} in')
fig.savefig(os.path.join(OUTDIR, 'panel_d_detergent_tolerance.pdf'),
            bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_d_detergent_tolerance.png'), dpi=300,
            bbox_inches='tight')

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
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
bars = pd.DataFrame([
    {'series': 'bar height (mean)' if label == 'protein groups'
     else 'line marker (condition)',
     'metric': METRIC[label], 'condition': cat, 'run': '',
     'replicate': np.nan, 'value': mean, 'pct_of_control': np.nan,
     'precursor_sum_raw_1e12': np.nan, 'median_pg_cv_pct': cv_by_cond[cat]}
    for label, cat, mean in heights])
# Peptide counts and the median peptide CV are what the text quotes, so they
# travel with the panel rather than only with the supplement.
extra = pd.DataFrame([
    {'series': 'condition summary', 'metric': m, 'condition': c, 'run': '',
     'replicate': np.nan, 'value': v, 'pct_of_control': np.nan,
     'precursor_sum_raw_1e12': np.nan, 'median_pg_cv_pct': cv_by_cond[c]}
    for c in ORDER
    for m, v in (('Peptides per run', pep.loc[c, 'peptides_per_run']),
                 ('Median peptide CV [%]',
                  round(pep.loc[c, 'median_peptide_cv_pct'], 2)))])
pd.concat([pts, bars[cols], extra[cols]], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'panel_d_detergent_tolerance_sourcedata.csv'),
    index=False)

print(f'Saved the detergent panel to {OUTDIR}')
