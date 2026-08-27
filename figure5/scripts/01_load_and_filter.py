"""Step 1 — Load the DIA-NN combined report, build sample_info, run the GMM"""
import re
import pandas as pd
import spec_analytics as core
import common as C
C.init()
WELL_RE = re.compile(r'_(EDL|soleus)_([A-Z]\d+)$')
experiments = [
    {'path': str(C.REPORT), 'file_tags': 'EDL',
     'condition1': 'EDL', 'condition2': 'fiber'},
    {'path': str(C.REPORT), 'file_tags': 'soleus',
     'condition1': 'SOL', 'condition2': 'fiber'},
]
df, sample_info = core.load_experiments(experiments)
sample_info['well'] = sample_info['run'].map(
    lambda r: WELL_RE.search(r).group(2))
sample_info['replicate'] = (sample_info['well'].str.extract(r'(\d+)')[0]
                            .astype(int))
print(f'loaded {len(df):,} precursor rows, {sample_info.shape[0]} runs')
print(sample_info['condition1'].value_counts().to_string())
pr_per_run = (df.groupby('run')['precursor_id'].nunique()
                .rename('n_precursors'))
df, sample_info, qc = core.filter_outlier_runs(
    df, sample_info,
    cohort_col='condition1',
    plot=True,
    figsize=(8, 4),
    palette=[C.MUSCLE_COLOR['EDL'], C.MUSCLE_COLOR['SOL']],
)
fig = C.plt.gcf()
C.unbold(*fig.axes)
thresholds = qc.set_index('cohort')['threshold']
sd = pd.DataFrame({'n_precursors': pr_per_run})
sd['cohort'] = ['EDL' if '_EDL_' in r else 'SOL' for r in sd.index]
sd['log10_n_precursors'] = C.np.log10(sd['n_precursors'])
sd['cohort_threshold'] = sd['cohort'].map(thresholds)
sd['kept'] = sd.index.isin(sample_info['run'])
sd = sd.reset_index().rename(columns={'index': 'run'})
C.save_panel(fig, 'supporting_outlier_filter_qc', sd)
qc.to_csv(C.DATA / 'outlier_summary.csv', index=False)
df.to_parquet(C.DF_FILTERED)
sample_info.to_parquet(C.SI_FILTERED)
print(f'\nAfter filter: {sample_info.shape[0]} fibers '
      f'({(sample_info["condition1"] == "EDL").sum()} EDL, '
      f'{(sample_info["condition1"] == "SOL").sum()} SOL)')
print('\nQC summary:')
print(qc.to_string(index=False))
