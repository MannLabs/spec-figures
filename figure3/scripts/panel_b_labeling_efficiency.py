"""Figure 3b — mTRAQ labelling efficiency per channel (H032_E229)."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
RAW = INPUT
OUTDIR = _cfg.output_dir(__file__)
os.makedirs(OUTDIR, exist_ok=True)

LABELLED = os.path.join(RAW, 'H032_E229.parquet')
LABELFREE = os.path.join(RAW, 'H032_E229_labelfree.parquet')

# Plate row -> mTRAQ channel of that run.
ROW_CHANNEL = {'B': '0', 'C': '4', 'D': '8'}
CHANNELS = ['0', '4', '8']
LABELS = {'0': 'd0', '4': 'd4', '8': 'd8'}
QVALUE = 0.01
N_REPLICATES = 3

BAR_WIDTH_IN = 0.38   # drawn bar width, matched across figure 3
AXES_H_IN = 2.79
POINT_SIZE = core.replicate_point_size(BAR_WIDTH_IN)

CHANNEL_COLOR = {'0': core.PALETTE_SINGLE[0],    # coral
                 '4': core.PALETTE_SINGLE[2],    # sky blue
                 '8': core.PALETTE_SINGLE[5]}    # pink

# ---------------------------------------------------------------------------
# Per-run labelled and unlabelled intensity.
# ---------------------------------------------------------------------------
lab = pd.read_parquet(LABELLED, columns=[
    'Run', 'Channel', 'Precursor.Quantity', 'Q.Value', 'Channel.Q.Value',
    'PG.Q.Value'])
lab['own_channel'] = lab['Run'].str[-2].map(ROW_CHANNEL)
if lab['own_channel'].isna().any():
    raise ValueError('unmapped plate rows in the labelled search')
lab = lab[(lab['Channel'] == lab['own_channel'])
          & (lab['Q.Value'] < QVALUE)
          & (lab['Channel.Q.Value'] < QVALUE)
          & (lab['PG.Q.Value'] < QVALUE)]

free = pd.read_parquet(LABELFREE, columns=[
    'Run', 'Precursor.Quantity', 'Q.Value', 'PG.Q.Value'])
free = free[(free['Q.Value'] < QVALUE) & (free['PG.Q.Value'] < QVALUE)]

labelled = lab.groupby('Run')['Precursor.Quantity'].sum()
unlabelled = free.groupby('Run')['Precursor.Quantity'].sum()

rows = []
for run in sorted(labelled.index):
    ch = ROW_CHANNEL[run[-2]]
    li, ui = labelled[run], unlabelled.get(run, 0.0)
    rows.append({'channel': ch, 'label': LABELS[ch], 'run': run,
                 'labelled_intensity': li, 'unlabelled_intensity': ui,
                 'efficiency_pct': 100 * li / (li + ui)})
per_run = pd.DataFrame(rows)
n_rep = per_run.groupby('channel').size()
if not (n_rep == N_REPLICATES).all():
    raise ValueError(f'expected {N_REPLICATES} replicates per channel, got {n_rep.to_dict()}')

means = per_run.groupby('channel')['efficiency_pct'].agg(['mean', 'std']).reindex(CHANNELS)
print('Labelling efficiency [%], intensity-weighted:')
print(means.round(2).to_string())

def set_axes_height_inches(fig, ax, h_in=AXES_H_IN):
    """Resize the figure so the axes rectangle is exactly `h_in` inches tall.

    Call after `tight_layout()`: the margins it measured are converted to inches
    and kept, so only the data area changes and the tick labels keep their
    clearance. Scaling the whole figure instead would shrink those margins while
    the text stayed fixed in points, which is how labels start colliding.
    """
    fig.canvas.draw()
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()
    bottom_in, top_in = pos.y0 * fig_h, (1.0 - pos.y1) * fig_h
    new_fig_h = bottom_in + h_in + top_in
    fig.set_size_inches(fig_w, new_fig_h)
    ax.set_position([pos.x0, bottom_in / new_fig_h,
                     pos.width, h_in / new_fig_h])


def set_bar_width_inches(fig, ax, target_in=BAR_WIDTH_IN):
    """Rescale bars so their drawn width is `target_in` inches.

    Bar width in data units maps to a different physical width in every panel,
    because the axes width depends on how wide the tick labels are. Setting it
    from the rendered geometry keeps the bars identical across panels.
    """
    fig.canvas.draw()
    ax_w = ax.get_position().width * fig.get_size_inches()[0]
    x0, x1 = ax.get_xlim()
    w_data = target_in * (x1 - x0) / ax_w
    for patch in ax.patches:
        centre = patch.get_x() + patch.get_width() / 2
        patch.set_width(w_data)
        patch.set_x(centre - w_data / 2)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
x = np.arange(len(CHANNELS))
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(2.75, 4))

for i, ch in enumerate(CHANNELS):
    ax.bar(i, means.loc[ch, 'mean'], 0.6, color=CHANNEL_COLOR[ch],
           edgecolor='black', linewidth=0.5, zorder=2)
    vals = per_run.loc[per_run['channel'] == ch, 'efficiency_pct'].to_numpy()
    jit = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jit, vals, s=POINT_SIZE, color='black',
               alpha=0.75, linewidth=0.3, edgecolor='white', zorder=5)
    ax.text(i, vals.max() + 1.4, f"{means.loc[ch, 'mean']:.1f}", ha='center',
            va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in CHANNELS], fontsize=8)
ax.set_xlim(-0.6, len(CHANNELS) - 0.4)
ax.set_ylim(0, 105)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel('Labelling efficiency [%]', fontsize=8)
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
set_axes_height_inches(fig, ax)
set_bar_width_inches(fig, ax)
print(f'axes {ax.get_position().width * fig.get_size_inches()[0]:.2f} x '
      f'{ax.get_position().height * fig.get_size_inches()[1]:.2f} in, '
      f'figure {fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in')
fig.savefig(os.path.join(OUTDIR, 'panel_b_labeling_efficiency.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_b_labeling_efficiency.png'), dpi=300,
            bbox_inches='tight')

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------
pts = per_run.assign(series='replicate point')[
    ['series', 'label', 'channel', 'run', 'labelled_intensity',
     'unlabelled_intensity', 'efficiency_pct']]
bars = pd.DataFrame([
    {'series': 'bar height (mean)', 'label': LABELS[c], 'channel': c, 'run': '',
     'labelled_intensity': np.nan, 'unlabelled_intensity': np.nan,
     'efficiency_pct': means.loc[c, 'mean']} for c in CHANNELS])
pd.concat([pts, bars], ignore_index=True).to_csv(
    os.path.join(OUTDIR, 'panel_b_labeling_efficiency_sourcedata.csv'), index=False)

print(f'\nSaved panel b to {OUTDIR}')
