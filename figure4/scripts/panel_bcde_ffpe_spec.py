"""Figure 4b-e — FFPE mouse liver, SPEC vs ISD+ (H032_E127)."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spec_analytics as core
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
OUTDIR = _cfg.output_dir(__file__)
os.makedirs(OUTDIR, exist_ok=True)
QVALUE = 0.01
MIN_REPS_FOR_CV = 3
REPORTS = {'Single-shot SPEC': os.path.join(INPUT, 'H032_E127',
                                            'single-shot_SPEC', 'report.parquet'),
           'Single-shot ISD+': os.path.join(INPUT, 'H032_E127',
                                            'single-shot_ISD+', 'report.parquet'),
           'Bulk ISD+': os.path.join(INPUT, 'H032_E127',
                                     'bulk_ISD+', 'report.parquet')}
ORDER = ['Single-shot ISD+', 'Bulk ISD+', 'Single-shot SPEC']
def lighten(color, factor=0.45):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * factor)
COLOR = {'Single-shot SPEC': core.PALETTE_SINGLE[0],
         'Bulk ISD+': core.PALETTE_SINGLE[5],
         'Single-shot ISD+': lighten(core.PALETTE_SINGLE[5])}
XTICK = {'Single-shot ISD+': 'Single-shot\nISD+', 'Bulk ISD+': 'Bulk\nISD+',
         'Single-shot SPEC': 'Single-shot\nSPEC'}
REPLICATE_TYPE = {'Single-shot ISD+': '4 digests', 'Bulk ISD+': '4 injections',
                  'Single-shot SPEC': '4 digests'}
FONTSIZE = 8
BAR_WIDTH_IN = 0.38
SHOW_VALUE_LABELS = False
POINT_SIZE = core.replicate_point_size(BAR_WIDTH_IN)
frames = []
for condition, path in REPORTS.items():
    d = pd.read_parquet(path, columns=[
        'Run', 'Protein.Group', 'PG.MaxLFQ', 'Precursor.Id', 'Precursor.Quantity',
        'Decoy', 'Q.Value', 'PG.Q.Value'])
    d = d[(d['Decoy'] == 0) & (d['Q.Value'] < QVALUE)
          & (d['PG.Q.Value'] < QVALUE) & (d['Precursor.Quantity'] > 0)]
    frames.append(d.assign(condition=condition))
raw = pd.concat(frames, ignore_index=True)
raw['tag'] = raw['Run'].str[-2:]
df = pd.DataFrame({
    'run': raw['Run'].astype(str),
    'engine': 'diann',
    'protein_group': raw['Protein.Group'].astype(str),
    'pg_intensity': raw['PG.MaxLFQ'].astype(float).replace(0, np.nan),
}).drop_duplicates(['run', 'protein_group'])
long = pd.DataFrame({
    'run': raw['Run'].astype(str),
    'protein_group': raw['Protein.Group'].astype(str),
    'precursor_id': raw['Precursor.Id'].astype(str),
    'precursor_intensity': raw['Precursor.Quantity'].astype(float),
})
long['pg_intensity'] = core.compute_directlfq_pg_intensity(long, num_cores=1)
long['pg_intensity'] = long['pg_intensity'].replace(0, np.nan)
df_joint = (long[['run', 'protein_group', 'pg_intensity']]
            .assign(engine='diann')
            .drop_duplicates(['run', 'protein_group']))
sample_info = (raw[['Run', 'tag', 'condition']].drop_duplicates()
               .rename(columns={'Run': 'run', 'condition': 'condition2'}))
sample_info['engine'] = 'diann'
sample_info['replicate'] = sample_info['tag'].str[1].astype(int)
counts = (df.dropna(subset=['pg_intensity'])
          .merge(sample_info[['run', 'condition2', 'replicate']], on='run')
          .groupby(['condition2', 'replicate'])['protein_group'].nunique()
          .rename('protein_groups').reset_index())
print('protein groups per run:')
print(counts.pivot(index='replicate', columns='condition2',
                   values='protein_groups')[ORDER].to_string())
def restyle(ax, fig, filename, ylabel=None, xlabel=None, annotations=False,
            rasterize_points=False):
    """House font sizes; core bolds axis labels, this figure does not.
    `rasterize_points` flattens the point clouds of d and e into a single 600 dpi
    image inside the PDF. Drawn as vectors they are ~7,000 individually coloured
    paths per panel, which is what slows Illustrator down; axes, ticks, labels and
    the annotation stay vector and editable.
    """
    for axis_label in (ax.xaxis.label, ax.yaxis.label):
        axis_label.set_fontweight('normal')
        axis_label.set_fontsize(FONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    if annotations:
        for text in ax.texts:
            text.set_fontsize(7)
            if text.get_bbox_patch() is not None:
                text.set_bbox(dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.set_title('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if rasterize_points:
        for collection in ax.collections:
            collection.set_rasterized(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{filename}.pdf'), bbox_inches='tight',
                dpi=600 if rasterize_points else None)
    fig.savefig(os.path.join(OUTDIR, f'{filename}.png'), dpi=300, bbox_inches='tight')
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(2.2, 2.6))
for i, cond in enumerate(ORDER):
    vals = counts.loc[counts['condition2'] == cond, 'protein_groups'].to_numpy()
    ax.bar(i, vals.mean(), 0.6, color=COLOR[cond], edgecolor='black',
           linewidth=0.5, zorder=2)
    ax.scatter(np.full(len(vals), i) + rng.uniform(-0.09, 0.09, len(vals)), vals,
               s=POINT_SIZE, color='black', alpha=0.75, linewidth=0.3, edgecolor='white',
               zorder=5)
    if SHOW_VALUE_LABELS:
        ax.text(i, vals.max() * 1.03, f'{vals.mean():,.0f}', ha='center',
                va='bottom', fontsize=7.5, fontweight='bold')
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([XTICK[c] for c in ORDER], fontsize=FONTSIZE, rotation=45,
                   ha='right')
ax.set_xlim(-0.6, len(ORDER) - 0.4)
ax.set_ylim(0, counts['protein_groups'].max() * 1.18)
ax.set_ylabel('Protein groups', fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
fig.canvas.draw()
ax_w = ax.get_position().width * fig.get_size_inches()[0]
w_data = BAR_WIDTH_IN * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax_w
for patch in ax.patches:
    centre = patch.get_x() + patch.get_width() / 2
    patch.set_width(w_data)
    patch.set_x(centre - w_data / 2)
restyle(ax, fig, 'panel_b_protein_groups')
counts.assign(series='replicate point').to_csv(
    os.path.join(OUTDIR, 'panel_b_protein_groups_sourcedata.csv'), index=False)
fig, ax, cv_stats = core.plot_cv_violin(
    df, sample_info, level='protein', group_col='condition2', group_order=ORDER,
    palette=COLOR, as_percent=True, ylim=(0, 60), figsize=(2.2, 2.6),
    min_values_for_cv=MIN_REPS_FOR_CV, y_label='Protein group CV [%]',
    median_label_loc='none', show_threshold=True, legend_fontsize=7)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([XTICK[c] for c in ORDER], fontsize=FONTSIZE, rotation=45,
                   ha='right')
restyle(ax, fig, 'panel_c_cv')
print('\nCV statistics from core.plot_cv_violin:')
print(cv_stats.to_string(index=False))
cv_table = core._compute_cv_table(df, sample_info, level='protein',
                                  group_col='condition2',
                                  min_values_for_cv=MIN_REPS_FOR_CV)
cv_table.assign(cv_pct=cv_table['cv'] * 100).to_csv(
    os.path.join(OUTDIR, 'panel_c_cv_sourcedata.csv'), index=False)
fig, ax, plot_df = core.plot_correlation(
    df_joint, sample_info, level='protein', group_col='condition2',
    condition_a='Bulk ISD+', condition_b='Single-shot SPEC',
    figsize=(2.2, 2.5), point_size=4, alpha=0.7, color_by_density=True,
    show_diagonal=True)
restyle(ax, fig, 'panel_d_spec_vs_bulk', annotations=True, rasterize_points=True,
        xlabel=r'$\log_2$ intensity, bulk ISD+',
        ylabel=r'$\log_2$ intensity, SPEC')
plot_df.to_csv(os.path.join(OUTDIR, 'panel_d_spec_vs_bulk_sourcedata.csv'),
               index=False)
print(f'\npanel d: n = {len(plot_df):,} protein groups')
rep_info = sample_info[sample_info['tag'].isin(['A1', 'A2'])].copy()
rep_info['condition2'] = 'SPEC replicate ' + rep_info['tag'].str[1]
fig, ax, plot_df = core.plot_correlation(
    df[df['run'].isin(rep_info['run'])], rep_info, level='protein',
    group_col='condition2', condition_a='SPEC replicate 1',
    condition_b='SPEC replicate 2', figsize=(2.2, 2.5), point_size=4, alpha=0.7,
    color_by_density=True, show_diagonal=True)
restyle(ax, fig, 'panel_e_spec_replicates', annotations=True,
        rasterize_points=True,
        xlabel=r'$\log_2$ intensity, SPEC replicate 1',
        ylabel=r'$\log_2$ intensity, SPEC replicate 2')
plot_df.to_csv(os.path.join(OUTDIR, 'panel_e_spec_replicates_sourcedata.csv'),
               index=False)
print(f'panel e: n = {len(plot_df):,} protein groups')
print(f'\nSaved panels b-e to {OUTDIR}')
