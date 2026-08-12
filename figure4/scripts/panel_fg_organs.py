"""Figure 4f + 4g — FFPE mouse organs, SPEC vs PAC (H032_E170)."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
REPORTS = {'SPEC': os.path.join(INPUT, 'H032_E170', 'SPEC', 'report.parquet'),
           'PAC': os.path.join(INPUT, 'H032_E170', 'PAC', 'report.parquet')}
OUTDIR = _cfg.output_dir(__file__)
os.makedirs(OUTDIR, exist_ok=True)

QVALUE = 0.01
MIN_REPS_FOR_CV = 3
ROW_ORGAN = {'A': 'Liver', 'B': 'Brain', 'C': 'Heart', 'D': 'Kidney',
             'E': 'Testis', 'F': 'Lung'}
# PAC first, so it sits left of SPEC in both panels.
METHODS = ['PAC', 'SPEC']
COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2]}
DODGE = {'PAC': -0.16, 'SPEC': 0.16}

FONTSIZE = 8

# ---------------------------------------------------------------------------
# Load; keep protein-group intensities for the CV panel.
# ---------------------------------------------------------------------------
frames = []
for method, path in REPORTS.items():
    available = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = [c for c in ('Run', 'Protein.Group', 'PG.MaxLFQ', 'Decoy', 'Q.Value',
                          'PG.Q.Value') if c in available]
    d = pd.read_parquet(path, columns=wanted)
    mask = (d['Q.Value'] < QVALUE) & (d['PG.Q.Value'] < QVALUE)
    if 'Decoy' in d:
        mask &= d['Decoy'] == 0
    else:
        print(f'note: no Decoy column in the {method} report; q-value filters only')
    frames.append(d[mask].assign(method=method))
raw = pd.concat(frames, ignore_index=True)

raw['tag'] = raw['Run'].str.extract(r'_([A-G]\d{1,2})$')[0]
raw['row'] = raw['tag'].str[0]
raw['organ'] = raw['row'].map(ROW_ORGAN)
if raw['organ'].isna().any():
    raise ValueError('runs whose plate row maps to no organ: '
                     f'{sorted(raw.loc[raw.organ.isna(), "tag"].unique())}')

# Canonical (df, sample_info) for core.plot_cv_violin.
df = pd.DataFrame({
    'run': raw['Run'].astype(str),
    'engine': 'diann',
    'protein_group': raw['Protein.Group'].astype(str),
    'pg_intensity': raw['PG.MaxLFQ'].astype(float).replace(0, np.nan),
}).drop_duplicates(['run', 'protein_group'])
sample_info = (raw[['Run', 'tag', 'organ', 'method']].drop_duplicates()
               .rename(columns={'Run': 'run'}))
sample_info['engine'] = 'diann'

counts = (df.dropna(subset=['pg_intensity'])
          .merge(sample_info[['run', 'organ', 'method']], on='run')
          .groupby(['organ', 'method', 'run'])['protein_group'].nunique()
          .rename('protein_groups').reset_index())

means = counts.pivot_table(index='organ', columns='method',
                           values='protein_groups', aggfunc='mean')
ORGAN_ORDER = list(means['SPEC'].sort_values(ascending=False).index)
means = means.reindex(ORGAN_ORDER)
means['SPEC - PAC'] = means['SPEC'] - means['PAC']
print('\nruns per organ and method:')
print(counts.pivot_table(index='organ', columns='method', values='run',
                         aggfunc='count').reindex(ORGAN_ORDER).to_string())
print('\nmean protein groups (organs ordered by SPEC):')
print(means.round(0).to_string())

# ---------------------------------------------------------------------------
# f — protein groups per organ
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
xpos = {organ: i for i, organ in enumerate(ORGAN_ORDER)}

fig, ax = plt.subplots(figsize=(4.0, 2.6))
for method in METHODS:
    sub = counts[counts['method'] == method]
    x = (sub['organ'].map(xpos) + DODGE[method]
         + rng.uniform(-0.05, 0.05, size=len(sub)))
    ax.scatter(x, sub['protein_groups'], s=18, color=COLOR[method], alpha=0.85,
               edgecolor='black', linewidth=0.4, label=method, zorder=4)

ax.set_xticks(range(len(ORGAN_ORDER)))
ax.set_xticklabels(ORGAN_ORDER, fontsize=FONTSIZE, rotation=45, ha='right')
ax.set_xlim(-0.5, len(ORGAN_ORDER) - 0.5)
ax.set_ylim(0, counts['protein_groups'].max() * 1.12)
ax.set_ylabel('Protein groups', fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', frameon=False, fontsize=FONTSIZE, handlelength=0.9,
          handletextpad=0.4, borderpad=0.2, ncol=2, columnspacing=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'panel_f_organs.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_f_organs.png'), dpi=300, bbox_inches='tight')

counts.sort_values(['organ', 'method'])[
    ['organ', 'method', 'run', 'protein_groups']].to_csv(
    os.path.join(OUTDIR, 'panel_f_organs_sourcedata.csv'), index=False)

# ---------------------------------------------------------------------------
# g — protein-group CV per organ, split by method (core.plot_cv_violin)
# ---------------------------------------------------------------------------
fig, ax, cv_stats = core.plot_cv_violin(
    df, sample_info, level='protein', group_col='organ', hue_col='method',
    group_order=ORGAN_ORDER, palette=COLOR, as_percent=True, ylim=(0, 40),
    figsize=(4.0, 2.6), min_values_for_cv=MIN_REPS_FOR_CV,
    y_label='Protein group CV [%]', show_median_label=False,
    show_threshold=True, legend_fontsize=7)
handles, labels = ax.get_legend_handles_labels()
if ax.get_legend() is not None:
    ax.get_legend().remove()
ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.0),
          ncol=len(labels), frameon=False, fontsize=7, handlelength=1.0,
          handleheight=0.9, handletextpad=0.4, borderpad=0.0, columnspacing=1.0)
ax.set_xticks(range(len(ORGAN_ORDER)))
ax.set_xticklabels(ORGAN_ORDER, fontsize=FONTSIZE, rotation=45, ha='right')
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_fontweight('normal')
ax.yaxis.label.set_fontsize(FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'panel_g_organ_cv.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_g_organ_cv.png'), dpi=300,
            bbox_inches='tight')

print('\nCV per organ and method [%]:')
print(cv_stats.assign(median_CV_pct=(cv_stats['median_CV'] * 100).round(1))
      [['group', 'hue', 'n_total', '%_CV<20%', 'median_CV_pct']].to_string(index=False))

cv_table = core._compute_cv_table(df, sample_info, level='protein',
                                  group_col='organ', hue_col='method',
                                  min_values_for_cv=MIN_REPS_FOR_CV)
cv_table.assign(cv_pct=cv_table['cv'] * 100).to_csv(
    os.path.join(OUTDIR, 'panel_g_organ_cv_sourcedata.csv'), index=False)

print(f'\nSaved panels f and g to {OUTDIR}')
