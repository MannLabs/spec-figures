"""Figure 1f — protein-group agreement between the two LC front ends (H032_E333)."""

import itertools
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__, 'H032_E333')
OUTDIR = _cfg.output_dir(__file__)
CACHE = _cfg.data_dir(__file__, 'lc_joint_directlfq_log2.parquet')

FRONT_ENDS = [
    ('Online Trap&Elute', os.path.join(INPUT, 'Vanquish', 'Vanquish.parquet')),
    ('Disposable Trap Column', os.path.join(INPUT, 'Evosep', 'Evosep.parquet')),
]
QVALUE = 0.01
N_REPLICATES = 4
NUM_CORES = 6
RASTER_DPI = 600
FONTSIZE = 8
POINT_SIZE = 3
DENSITY_CMAP = 'inferno'
STEM = 'panel_f_lc_correlation'
AXES_H_IN = 2.50


def joint_quantification():
    parts, runs = [], {}
    for tag, path in FRONT_ENDS:
        d = pd.read_parquet(path, columns=[
            'Run', 'Precursor.Id', 'Protein.Group', 'Precursor.Quantity',
            'Q.Value', 'PG.Q.Value'])
        d = d[(d['Q.Value'] < QVALUE) & (d['PG.Q.Value'] < QVALUE)
              & (d['Precursor.Quantity'] > 0)]
        runs[tag] = sorted(d['Run'].unique())
        if len(runs[tag]) != N_REPLICATES:
            raise ValueError(f'{tag}: expected {N_REPLICATES} runs, '
                             f'found {len(runs[tag])}')
        parts.append(pd.DataFrame({
            'run': d['Run'].astype(str),
            'precursor_id': d['Precursor.Id'].astype(str),
            'protein_group': d['Protein.Group'].astype(str),
            'precursor_intensity': d['Precursor.Quantity'].astype(float)}))

    df = pd.concat(parts, ignore_index=True)
    print(f'pooled matrix: {df["run"].nunique()} runs, '
          f'{df["precursor_id"].nunique():,} precursors, '
          f'{df["protein_group"].nunique():,} protein groups')

    if os.path.exists(CACHE):
        print(f'reusing {os.path.relpath(CACHE, OUTDIR)}')
        return pd.read_parquet(CACHE), runs

    df['pg_intensity'] = core.compute_directlfq_pg_intensity(
        df, num_cores=NUM_CORES)
    pg = (df.dropna(subset=['pg_intensity'])
          .drop_duplicates(['run', 'protein_group'])
          .pivot(index='protein_group', columns='run', values='pg_intensity'))
    lg = np.log2(pg.replace(0, np.nan))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    lg.to_parquet(CACHE)
    return lg, runs


def set_axes_size_inches(fig, ax, *, w_in=None, h_in=None):
    """Resize the figure so the axes rectangle is exactly `w_in` x `h_in`.

    Call after `tight_layout()`: the margins it measured are converted to inches
    and preserved, so only the data area changes size and the tick labels and
    axis labels keep the clearance they were given. `bbox_inches='tight'` crops
    without scaling, so the saved PDF keeps these inch values.
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


def main():
    lg, runs = joint_quantification()
    complete = lg.dropna()
    print(f'complete cases across all 8 runs: {len(complete):,} of {len(lg):,}')

    # Within-front-end references, reported but not plotted: they are what says
    # whether the cross-front-end spread is large or small.
    for tag, r in runs.items():
        w = complete[r]
        h1, h2 = w.iloc[:, :2].mean(axis=1), w.iloc[:, 2:].mean(axis=1)
        d = (h2 - h1).to_numpy()
        print(f'within {tag:24s} r = {stats.pearsonr(h1, h2)[0]:.4f}  '
              f'IQR = {np.subtract(*np.percentile(d, [75, 25])):.3f}')

    x = complete[runs['Online Trap&Elute']].mean(axis=1)
    y = complete[runs['Disposable Trap Column']].mean(axis=1)
    d = (y - x).to_numpy()
    pearson = stats.pearsonr(x, y)[0]
    spearman = stats.spearmanr(x, y)[0]
    iqr = float(np.subtract(*np.percentile(d, [75, 25])))
    print(f'\nacross front ends, n = {len(x):,}')
    print(f'  r = {pearson:.4f}  rho = {spearman:.4f}')
    print(f'  log2 difference: median {np.median(d):+.3f}, IQR {iqr:.3f}, '
          f'within +/-1 for {np.mean(np.abs(d) < 1):.1%}')
    q = pd.qcut(x, 5, labels=False)
    print('  r by abundance quintile: ' + '  '.join(
        f'Q{k + 1} {stats.pearsonr(x[q == k], y[q == k])[0]:.2f}' for k in range(5)))

    xv, yv = x.to_numpy(), y.to_numpy()
    pad = 0.03 * (max(xv.max(), yv.max()) - min(xv.min(), yv.min()))
    limits = (min(xv.min(), yv.min()) - pad, max(xv.max(), yv.max()) + pad)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    try:
        density = gaussian_kde(np.vstack([xv, yv]))(np.vstack([xv, yv]))
    except np.linalg.LinAlgError:
        density = np.ones_like(xv)
    order = density.argsort()
    cloud = ax.scatter(xv[order], yv[order], c=density[order], s=POINT_SIZE,
                       alpha=0.5, cmap=DENSITY_CMAP, edgecolor='none')
    cloud.set_rasterized(True)
    ax.plot(limits, limits, color='black', linestyle='--', linewidth=0.7,
            zorder=3)

    ax.text(0.04, 0.96,
            f'r = {pearson:.3f}\nρ = {spearman:.3f}\nn = {len(x):,}',
            transform=ax.transAxes, fontsize=FONTSIZE, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.85), zorder=6)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal')
    ax.set_xlabel('log₂ intensity, online trap&elute', fontsize=FONTSIZE)
    ax.set_ylabel('log₂ intensity, disposable trap column', fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    # Pin the square data area rather than leaving it to figsize minus whatever
    # margins tight_layout measured, so it matches c, d and e to the inch.
    set_axes_size_inches(fig, ax, w_in=AXES_H_IN, h_in=AXES_H_IN)
    fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight',
                dpi=RASTER_DPI)
    fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300,
                bbox_inches='tight')

    pd.DataFrame({
        'protein_group': x.index,
        'log2_online_trap_and_elute': xv,
        'log2_disposable_trap_column': yv,
        'log2_difference': d,
    }).to_csv(os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)

    print(f'\nSaved panel f to {OUTDIR}')


if __name__ == '__main__':
    main()
