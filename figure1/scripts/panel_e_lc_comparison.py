"""Figure 1e — one SAX SPEC digest on two LC front ends (H032_E333)."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import spec_analytics as core
from spec_analytics import load_experiments

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__, 'H032_E333')
OUTDIR = _cfg.output_dir(__file__)

TRAP_COLUMN_COLOR = core.PALETTE_SINGLE[0]   # coral, as everywhere else
ONLINE_COLOR = core.PALETTE_SINGLE[4]        # yellow, unused elsewhere in fig. 1

CONDITIONS = [
    ('Vanquish', 'Online\nTrap&Elute',      ['_A5', '_A6', '_A7', '_A8'],
     ONLINE_COLOR),
    ('Evosep',   'Disposable\nTrap Column', ['_A1', '_A2', '_A3', '_A4'],
     TRAP_COLUMN_COLOR),
]
ORDER = [key for key, _label, _tags, _color in CONDITIONS]
LABEL = {key: label for key, label, _tags, _color in CONDITIONS}
COLOR = {key: color for key, _label, _tags, color in CONDITIONS}

MIN_VALUES_FOR_CV = 3
SHOW_VALUE_LABELS = False
BAR_IN = 0.38
POINT_SIZE = core.replicate_point_size(BAR_IN)
UNIT_IN = 0.50
# Data-area height shared by every panel of the figure; see the same constant in
# panel_cd_phase_comparison.py for why it is 2.50 in.
AXES_H_IN = 2.50
FONTSIZE = 8
# 8 pt is the house floor for any text; the key used to sit at 7 pt.
LEGEND_FONTSIZE = 8
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


def set_axes_size_inches(fig, ax, *, w_in=None, h_in=None):
    """Resize the figure so the axes rectangle is exactly `w_in` x `h_in`.

    Call after `tight_layout()`: the margins it measured are converted to inches
    and preserved, so only the data area changes size and the tick labels, axis
    labels and legend keep the clearance they were given. `bbox_inches='tight'`
    crops without scaling, so the saved PDF keeps these inch values. Either
    dimension may be left as it is by passing None.
    """
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    left_in, right_in = pos.x0 * fig_w, (1.0 - pos.x1) * fig_w
    bottom_in, top_in = pos.y0 * fig_h, (1.0 - pos.y1) * fig_h
    axes_w = pos.width * fig_w if w_in is None else w_in
    axes_h = pos.height * fig_h if h_in is None else h_in

    new_fig_w = left_in + axes_w + right_in
    new_fig_h = bottom_in + axes_h + top_in
    fig.set_size_inches(new_fig_w, new_fig_h)
    ax.set_position([left_in / new_fig_w, bottom_in / new_fig_h,
                     axes_w / new_fig_w, axes_h / new_fig_h])


def set_bar_geometry_inches(fig, ax, *, bar_in=BAR_IN, unit_in=UNIT_IN,
                            h_in=AXES_H_IN):
    """Resize the axes so one x data unit is `unit_in`, then set bars to `bar_in`.

    Call after `tight_layout()`. Fixing only the bar width leaves the gap set by
    whatever axes width tight_layout happened to produce, so a 2-bar panel and a
    5-bar panel end up with visibly different spacing. Rescaling the axes first
    pins the pitch, which pins the gap. The height is pinned in the same call,
    because a bar panel and an ECDF panel otherwise inherit different data-area
    heights from the same figure height whenever their x labels differ in depth.
    """
    x0, x1 = ax.get_xlim()
    set_axes_size_inches(fig, ax, w_in=(x1 - x0) * unit_in, h_in=h_in)

    w_data = bar_in / unit_in
    for patch in ax.patches:
        centre = patch.get_x() + patch.get_width() / 2
        patch.set_width(w_data)
        patch.set_x(centre - w_data / 2)


# ---------------------------------------------------------------------------
# Load the two combined reports.
# ---------------------------------------------------------------------------
experiments = [
    {'path': os.path.join(INPUT, key, f'{key}.parquet'), 'file_tags': tags,
     'condition1': 'H032_E333', 'condition2': key}
    for key, _label, tags, _color in CONDITIONS
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

means = per_run.groupby('condition')['protein_groups'].agg(['mean', 'std', 'size'])
means = means.reindex(ORDER)
means['pg_cv20'] = [per_run.loc[per_run['condition'] == c, 'pg_cv20'].iloc[0]
                    for c in ORDER]
means['median_pg_cv_pct'] = [
    100 * cv_pg.loc[cv_pg['group'] == c, 'cv'].median() for c in ORDER]
print('protein groups per replicate (mean +/- SD), CV<20% subset, median CV:')
print(means.round(1).to_string())
delta = means['mean'].loc['Evosep'] / means['mean'].loc['Vanquish'] - 1
print(f'Evosep vs Vanquish: {delta:+.1%}')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
x = np.arange(len(ORDER))
rng = np.random.default_rng(0)
ymax = per_run['protein_groups'].max()

fig, ax = plt.subplots(figsize=(2.6, 4))
for i, cond in enumerate(ORDER):
    sub = per_run[per_run['condition'] == cond]
    total = float(sub['protein_groups'].mean())
    pg20 = float(sub['pg_cv20'].iloc[0])
    ax.bar(x[i], total, 0.6, color=lighten(COLOR[cond]), edgecolor='darkgray',
           linewidth=0.8, zorder=2)
    ax.bar(x[i], pg20, 0.6, color=COLOR[cond], edgecolor='black',
           linewidth=0.8, zorder=3)
    vals = sub['protein_groups'].to_numpy()
    jit = rng.uniform(-0.07, 0.07, size=len(vals))
    ax.scatter(np.full(len(vals), x[i]) + jit, vals, s=POINT_SIZE, color='black',
               alpha=0.75, linewidth=0.3, edgecolor='white', zorder=5)
    if SHOW_VALUE_LABELS:
        ax.text(x[i], max(total, vals.max()) + ymax * 0.022, f'{total:,.0f}',
                ha='center', va='bottom', fontsize=FONTSIZE)
        ax.text(x[i], pg20 - ymax * 0.012, f'{pg20:,.0f}', ha='center',
                va='top', fontsize=FONTSIZE, fontweight='bold',
                color=readable_on(COLOR[cond]))

ax.set_xticks(x)
ax.set_xticklabels([LABEL[c] for c in ORDER], fontsize=FONTSIZE,
                   rotation=45, ha='right')
ax.set_xlim(-0.6, len(ORDER) - 0.4)
ax.set_ylim(0, ymax * 1.12)
ax.set_ylabel('Protein groups', fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(
    handles=[Patch(facecolor=lighten(LEGEND_DARK, 0.55), edgecolor='darkgray',
                   linewidth=0.8, label='IDs'),
             Patch(facecolor=LEGEND_DARK, edgecolor='black', linewidth=0.8,
                   label='< 20% CV')],
    loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
    fontsize=LEGEND_FONTSIZE, handlelength=0.8, handleheight=0.9, handletextpad=0.35,
    borderpad=0.0, columnspacing=0.8)

fig.tight_layout()
set_bar_geometry_inches(fig, ax)
fig.savefig(os.path.join(OUTDIR, 'panel_e_lc_comparison.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_e_lc_comparison.png'), dpi=300,
            bbox_inches='tight')

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
cols = ['series', 'condition', 'label', 'run', 'replicate', 'protein_groups']
pts = per_run.assign(series='replicate point',
                     label=per_run['condition'].map(LABEL))
bars = []
for cond in ORDER:
    sub = per_run[per_run['condition'] == cond]
    bars.append({'series': 'bar height, IDs (mean)', 'condition': cond,
                 'label': LABEL[cond], 'run': '',
                 'replicate': np.nan,
                 'protein_groups': sub['protein_groups'].mean()})
    bars.append({'series': 'bar height, CV<20%', 'condition': cond,
                 'label': LABEL[cond], 'run': '',
                 'replicate': np.nan,
                 'protein_groups': float(sub['pg_cv20'].iloc[0])})
pd.concat([pts[cols], pd.DataFrame(bars)[cols]], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'panel_e_lc_comparison_sourcedata.csv'), index=False)

print(f'\nSaved panel e to {OUTDIR}')
