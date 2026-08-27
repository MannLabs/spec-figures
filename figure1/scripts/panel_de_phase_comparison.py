"""Figure 1c + 1d — sorbent phases and digestion formats (H032_E297)."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import spec_analytics as core
from spec_analytics import load_experiments
core.init_plotting()
PANEL_LEFT_IN = 0.81
def save_matched(fig, ax, stem):
    """Save PDF + PNG with the axes' left edge PANEL_LEFT_IN from the box edge."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    gutter = ax.get_position().x0 * fig.get_size_inches()[0] - tb.x0
    pad = max(0.0, PANEL_LEFT_IN - gutter)
    box = Bbox.from_extents(tb.x0 - pad, tb.y0, tb.x1, tb.y1)
    fig.savefig(os.path.join(OUTDIR, f'{stem}.pdf'), bbox_inches=box)
    fig.savefig(os.path.join(OUTDIR, f'{stem}.png'), dpi=300, bbox_inches=box)
    print(f'  {stem}: gutter {gutter:.3f} in, padded {pad:.3f}, '
          f'box {box.width:.3f} in')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__, 'H032_E297')
OUTDIR = _cfg.output_dir(__file__)
CONDITIONS = [
    ('C18',  'C18',         'C18',  'D', core.PALETTE_SINGLE[3]),
    ('ISD+', 'ISD_SDB-RPS', 'ISD+', 'A', core.PALETTE_SINGLE[5]),
    ('SCX',  'SCX',         'SCX',  'C', core.PALETTE_SINGLE[2]),
    ('ISD',  'ISD',         'ISD',  'E', core.PALETTE_SINGLE[1]),
    ('SAX',  'SAX',         'SAX',  'B', core.PALETTE_SINGLE[0]),
]
ORDER = [label for _f, _s, label, _l, _c in CONDITIONS]
COLOR = {label: color for _f, _s, label, _l, color in CONDITIONS}
MIN_VALUES_FOR_CV = 3
SHOW_VALUE_LABELS = False
BAR_IN = 0.195
POINT_SIZE = core.replicate_point_size(BAR_IN)
UNIT_IN = 0.26
AXES_H_IN = 2.50
AXES_W_IN = 2.60
FONTSIZE = 10
LEGEND_FONTSIZE = 10
LEGEND_DARK = '#595959'
def readable_on(color):
    """Black or white, whichever contrasts with `color` (Rec. 601 luminance).
    The `< 20% CV` count is printed inside the dark bar, so its colour has to
    follow the bar. Hard-coded white worked while every hue here was mid-tone, but
    it is illegible on the palette's yellow — computed rather than per-hue so it
    survives any future recolouring.
    """
    r, g, b = mcolors.to_rgb(color)
    return 'black' if 0.299 * r + 0.587 * g + 0.114 * b > 0.72 else 'white'
def lighten(color, factor=0.45):
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - ch) * (1 - factor) for ch in c)
experiments = [
    {'path': os.path.join(INPUT, folder, f'{stem}.parquet'),
     'file_tags': [f'_{letter}{i}' for i in range(1, 5)],
     'condition1': 'H032_E297', 'condition2': label}
    for folder, stem, label, letter, _color in CONDITIONS
]
df, sample_info = load_experiments(experiments, diann_pg_method='maxlfq')
cv_pg = core._compute_cv_table(df, sample_info, level='protein',
                               group_col='condition2',
                               min_values_for_cv=MIN_VALUES_FOR_CV)
pg_quant = df.dropna(subset=['pg_intensity'])
pg_quant = pg_quant[pg_quant['pg_intensity'] > 0]
rows = []
for cond in ORDER:
    runs = list(sample_info.loc[sample_info['condition2'] == cond, 'run'])
    pg20 = int((cv_pg.loc[cv_pg['group'] == cond, 'cv'].to_numpy() < 0.20).sum())
    for replicate, run in enumerate(sorted(runs), start=1):
        rows.append({
            'condition': cond, 'run': run, 'replicate': replicate,
            'protein_groups': int(
                pg_quant.loc[pg_quant['run'] == run, 'protein_group'].nunique()),
            'pg_cv20': pg20,
        })
per_run = pd.DataFrame(rows)
summary = per_run.groupby('condition')['protein_groups'].agg(
    ['mean', 'std', 'size']).reindex(ORDER)
summary['pg_cv20'] = [per_run.loc[per_run['condition'] == c, 'pg_cv20'].iloc[0]
                      for c in ORDER]
summary['pct_cv20'] = 100 * summary['pg_cv20'] / summary['mean']
summary['median_pg_cv_pct'] = [
    100 * cv_pg.loc[cv_pg['group'] == c, 'cv'].median() for c in ORDER]
summary['n_pg_for_cv'] = [int(cv_pg.loc[cv_pg['group'] == c, 'cv'].notna().sum())
                          for c in ORDER]
print('protein groups per replicate (mean +/- SD), CV<20% subset, median CV:')
print(summary.round(1).to_string())
print(f"SAX vs ISD:  {summary['mean'].loc['SAX'] / summary['mean'].loc['ISD'] - 1:+.1%}")
print(f"SAX vs C18:  {summary['mean'].loc['SAX'] / summary['mean'].loc['C18'] - 1:+.1%}")
GROUPS = [('in solution', ['ISD+', 'ISD']), ('SPEC', ['C18', 'SCX', 'SAX'])]
means = {c: float(per_run[per_run['condition'] == c]['protein_groups'].mean())
         for c in ORDER}
points = {c: per_run[per_run['condition'] == c]['protein_groups'].to_numpy()
          for c in ORDER}
ymax = per_run['protein_groups'].max()
fig, ax = plt.subplots(figsize=(3.2, 4))
core.plot_grouped_bars(
    GROUPS, means, colors={c: COLOR[c] for c in ORDER}, points=points,
    ax=ax, y_label='Protein groups', ylim=(0, ymax * 1.30),
    bar_in=BAR_IN, unit_in=UNIT_IN, legend=False,
    point_size=POINT_SIZE,
    tick_fontsize=FONTSIZE, label_fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: '0' if v == 0 else f'{v/1000:g}k'))
ax.tick_params(labelsize=FONTSIZE)
BAR_ORDER = [c for _g, members in GROUPS for c in members]
ax.legend(handles=[Patch(facecolor=COLOR[c], edgecolor='black', linewidth=0.7,
                         label=c) for c in BAR_ORDER],
          loc='upper center', ncol=3, frameon=False, fontsize=LEGEND_FONTSIZE,
          handlelength=0.9, handletextpad=0.4, columnspacing=0.9, borderpad=0.2)
fig.tight_layout()
core.fix_bar_geometry(fig, ax, bar_in=BAR_IN, unit_in=UNIT_IN, h_in=AXES_H_IN)
save_matched(fig, ax, 'panel_d_phase_protein_groups')
cols = ['series', 'group', 'condition', 'run', 'replicate', 'protein_groups']
group_of = {c: g for g, members in GROUPS for c in members}
pts = per_run.assign(series='replicate point',
                     group=per_run['condition'].map(group_of))
bars = [{'series': 'bar height (mean)', 'group': group_of[c], 'condition': c,
         'run': '', 'replicate': np.nan, 'protein_groups': means[c]}
        for c in ORDER]
pd.concat([pts[cols], pd.DataFrame(bars)[cols]], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'panel_d_phase_protein_groups_sourcedata.csv'),
    index=False)
fig, ax, cv_stats = core.plot_cv_ecdf(
    df, sample_info, level='protein', group_col='condition2',
    group_order=ORDER, palette=COLOR, min_values_for_cv=MIN_VALUES_FOR_CV,
    figsize=(3.5, 4), x_label='PG CV [%]', y_label='Cumulative fraction',
    legend_loc='lower right', legend_fontsize=FONTSIZE)
ax.xaxis.label.set_fontsize(FONTSIZE)
ax.yaxis.label.set_fontsize(FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
fig.tight_layout()
core.set_axes_size_inches(fig, ax, w_in=AXES_W_IN, h_in=AXES_H_IN)
fig.savefig(os.path.join(OUTDIR, 'panel_e_phase_cv_ecdf.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_e_phase_cv_ecdf.png'), dpi=300,
            bbox_inches='tight')
print('\nprotein-group CV summary:')
print(cv_stats.assign(median_CV_pct=(cv_stats['median_CV'] * 100).round(1))
      [['group', 'n_total', '%_CV<20%', 'median_CV_pct']].to_string(index=False))
ecdf = []
for cond in ORDER:
    vals = np.sort(cv_pg.loc[cv_pg['group'] == cond, 'cv'].dropna().to_numpy()) * 100
    ecdf.append(pd.DataFrame({
        'condition': cond, 'cv_pct': vals,
        'cumulative_fraction': np.arange(1, vals.size + 1) / vals.size}))
pd.concat(ecdf, ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'panel_e_phase_cv_ecdf_sourcedata.csv'), index=False)
print(f'\nSaved panels d and e to {OUTDIR}')
