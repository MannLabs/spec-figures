"""Does dropping the outlier filter change the single-fiber conclusions?"""
import os
import runpy
import sys
import pandas as pd
FIG5 = _cfg.output_dir_of('figure5')
SCRIPTS = os.path.join(FIG5, 'scripts')
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                       'sensitivity_no_outlier_filter')
os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(os.path.join(SCRATCH, 'plots'), exist_ok=True)
sys.path.insert(0, SCRIPTS)
os.chdir(SCRIPTS)
import spec_analytics as core

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
import common as C
from pathlib import Path
C.DATA = Path(SCRATCH)
for name in ('DF_FILTERED', 'SI_FILTERED', 'SI_TYPED'):
    if hasattr(C, name):
        setattr(C, name, Path(SCRATCH) / os.path.basename(str(getattr(C, name))))
if hasattr(C, 'PLOTS'):
    C.PLOTS = Path(SCRATCH) / 'plots'
if hasattr(C, 'OUT'):
    C.OUT = Path(SCRATCH)
C.save_panel = lambda *a, **k: None
def _no_filter(df, sample_info, *, cohort_col='condition1', **kwargs):
    """Keep every run, returning a QC frame with the columns step 01 goes on to read.
    `threshold` is -inf rather than NaN: the caller uses it to draw a cutoff line and
    to report how many runs fell below, and -inf keeps both truthful — nothing is
    below it, which is exactly what "no filter" means.
    """
    import numpy as np
    print(f'>>> outlier filter DISABLED: keeping all {sample_info.shape[0]} fibers')
    cohorts = sorted(sample_info[cohort_col].dropna().unique())
    qc = pd.DataFrame({
        'cohort': cohorts,
        'threshold': [-np.inf] * len(cohorts),
        'n_before': [int((sample_info[cohort_col] == c).sum()) for c in cohorts],
        'n_after': [int((sample_info[cohort_col] == c).sum()) for c in cohorts],
        'n_dropped': [0] * len(cohorts),
    })
    return df, sample_info, qc
core.filter_outlier_runs = _no_filter
for step in ('01_load_and_filter.py', '02_fiber_types.py', '14_volcano_I_vs_IIb.py'):
    print(f'\n{"=" * 70}\nRUNNING {step}\n{"=" * 70}')
    runpy.run_path(os.path.join(SCRIPTS, step), run_name='__main__')
filt = pd.read_csv(os.path.join(FIG5, 'data', 'volcano_I_vs_IIb.csv'))
unf = pd.read_csv(os.path.join(SCRATCH, 'volcano_I_vs_IIb.csv'))
def regulated(v):
    import numpy as np
    return set(v.loc[(v['p_adj'] < 0.05)
                     & (v['log2fc_I_vs_IIb'].abs() > np.log2(1.5)),
                     'protein_group'])
a, b = regulated(filt), regulated(unf)
print('\n' + '=' * 70)
print('SENSITIVITY TO THE OUTLIER FILTER')
print('=' * 70)
print(f'protein groups tested : {len(filt):,} filtered   {len(unf):,} unfiltered')
print(f'regulated             : {len(a):,} filtered   {len(b):,} unfiltered')
print(f'shared                : {len(a & b):,} '
      f'({100 * len(a & b) / len(a | b):.1f} % of the union)')
print(f'  filtered only       : {len(a - b):,}')
print(f'  unfiltered only     : {len(b - a):,}')
print(f'  recovered from the filtered set: {100 * len(a & b) / len(a):.1f} %')
m = filt.merge(unf, on='protein_group', suffixes=('_f', '_u'))
print(f'\nfold changes on the {len(m):,} protein groups tested in both: '
      f'Pearson r = {m["log2fc_I_vs_IIb_f"].corr(m["log2fc_I_vs_IIb_u"]):.4f}, '
      f'median |delta| = '
      f'{(m["log2fc_I_vs_IIb_f"] - m["log2fc_I_vs_IIb_u"]).abs().median():.3f} log2')
sign = ((m['log2fc_I_vs_IIb_f'] > 0) == (m['log2fc_I_vs_IIb_u'] > 0))
both = m[m['protein_group'].isin(a & b)]
sign_both = ((both['log2fc_I_vs_IIb_f'] > 0) == (both['log2fc_I_vs_IIb_u'] > 0))
print(f'direction preserved   : {100 * sign.mean():.2f} % of all tested, '
      f'{100 * sign_both.mean():.2f} % of the shared regulated set')
