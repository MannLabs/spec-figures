"""Supplementary figure 5c — PCA of the FFPE organ runs, SPEC vs PAC (H032_E170)."""
import os
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import pyarrow.parquet as pq
import spec_analytics as core
core.init_plotting()
PANEL_SIZE = (6.2, 3.2)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
REPORT = os.path.join(INPUT, 'H032_E170.parquet')
OUTDIR = _cfg.output_dir(__file__)
QVALUE = 0.01
ROW_ORGAN = {'A': 'Liver', 'B': 'Brain', 'C': 'Heart', 'D': 'Kidney',
             'E': 'Testis', 'F': 'Lung'}
ORGAN_ORDER = ['Testis', 'Kidney', 'Brain', 'Lung', 'Liver', 'Heart']
ORGAN_COLOR = {organ: core.PALETTE_SINGLE[i] for i, organ in enumerate(ORGAN_ORDER)}
MARKER = {'SPEC': 'o', 'PAC': 's'}
FONTSIZE = 8
FIG4_INPUT = _cfg.cross_input('figure4')
REPORTS = {
    'SPEC': os.path.join(FIG4_INPUT, 'H032_E170', 'SPEC', 'report.parquet'),
    'PAC': os.path.join(FIG4_INPUT, 'H032_E170', 'PAC', 'report.parquet')}
frames = []
for method, path in REPORTS.items():
    available = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = [c for c in ('Run', 'Protein.Group', 'Precursor.Id',
                          'Precursor.Quantity', 'Decoy', 'Q.Value',
                          'PG.Q.Value') if c in available]
    d = pd.read_parquet(path, columns=wanted)
    mask = ((d['Q.Value'] < QVALUE) & (d['PG.Q.Value'] < QVALUE)
            & (d['Precursor.Quantity'] > 0))
    if 'Decoy' in d:
        mask &= d['Decoy'] == 0
    frames.append(d[mask].assign(method=method))
raw = pd.concat(frames, ignore_index=True)
raw['tag'] = raw['Run'].str.extract(r'_([A-G]\d{1,2})$')[0]
raw['organ'] = raw['tag'].str[0].map(ROW_ORGAN)
if raw['organ'].isna().any():
    raise ValueError('runs whose plate row maps to no organ')
long = pd.DataFrame({
    'run': raw['Run'].astype(str),
    'protein_group': raw['Protein.Group'].astype(str),
    'precursor_id': raw['Precursor.Id'].astype(str),
    'precursor_intensity': raw['Precursor.Quantity'].astype(float),
})
long['pg_intensity'] = core.compute_directlfq_pg_intensity(long, num_cores=1)
long['pg_intensity'] = long['pg_intensity'].replace(0, np.nan)
df = (long[['run', 'protein_group', 'pg_intensity']]
      .assign(engine='diann')
      .drop_duplicates(['run', 'protein_group']))
sample_info = (raw[['Run', 'organ', 'method']].drop_duplicates()
               .rename(columns={'Run': 'run'}))
sample_info['engine'] = 'diann'
sample_info['replicate'] = sample_info.groupby(['organ', 'method']).cumcount() + 1
print(f'{len(sample_info)} runs, {df["protein_group"].nunique():,} protein groups')
fig, ax, res = core.plot_pca(
    df, sample_info, level='protein', color_by='organ', style_by='method',
    palette=ORGAN_COLOR, figsize=PANEL_SIZE, point_size=26, show_labels=False,
    show_ellipses=False, legend_fontsize=7, label_fontsize=FONTSIZE,
    tick_fontsize=FONTSIZE)
handles = [Line2D([], [], linestyle='none', marker='o', markersize=5,
                  markerfacecolor=ORGAN_COLOR[organ], markeredgecolor='black',
                  markeredgewidth=0.4, label=organ) for organ in ORGAN_ORDER]
handles += [Line2D([], [], linestyle='none', marker=MARKER[m], markersize=5,
                   markerfacecolor='#BBBBBB', markeredgecolor='black',
                   markeredgewidth=0.4, label=m) for m in ('SPEC', 'PAC')]
ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=7,
          handletextpad=0.4, labelspacing=0.25, borderpad=0.2, ncol=2,
          columnspacing=0.8)
for axis_label in (ax.xaxis.label, ax.yaxis.label):
    axis_label.set_fontweight('normal')
    axis_label.set_fontsize(FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.set_title('')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'suppl5_c_pca_organs.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'suppl5_c_pca_organs.png'), dpi=300,
            bbox_inches='tight')
scores = res['transformed']
if 'organ' not in scores:
    scores = scores.merge(sample_info[['run', 'organ', 'method']], on='run')
var = res['variance_explained']
print('\nvariance explained [%]:',
      ', '.join(f'PC{i + 1} {100 * v:.1f}' for i, v in enumerate(var[:4])))
def r_squared(values, groups):
    """Fraction of variance in `values` explained by a categorical grouping."""
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2
                     for _, g in values.groupby(groups, observed=True))
    return ss_between / ss_total if ss_total else np.nan
print('\nR^2 of each PC against organ and against method:')
for pc in [c for c in scores.columns if c.startswith('PC')][:4]:
    print(f'  {pc} ({100 * var[int(pc[2:]) - 1]:.1f}% of variance): '
          f'organ {r_squared(scores[pc], scores["organ"]):.3f}, '
          f'method {r_squared(scores[pc], scores["method"]):.3f}')
scores.to_csv(os.path.join(OUTDIR, 'suppl5_c_pca_organs_sourcedata.csv'), index=False)
print(f'\nSaved panel c to {OUTDIR}')
