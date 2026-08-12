"""Supplementary figure 2 — SAX SPEC reproduced two months apart."""

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
INPUT = _cfg.cross_input('supplementary_figure1')
OUTDIR = _cfg.output_dir(__file__)
CACHE = _cfg.data_dir(__file__, 'joint_directlfq_log2.parquet')

PREPARATIONS = [
    ('April', os.path.join(INPUT, 'H032_E297', 'SAX', 'SAX.parquet')),
    ('June', os.path.join(INPUT, 'H032_E333', 'Evosep', 'Evosep.parquet')),
]
QVALUE = 0.01
N_REPLICATES = 4
NUM_CORES = 6
RASTER_DPI = 600
FONTSIZE = 8
LABEL_FONTSIZE = 7.5
POINT_SIZE = 3
DENSITY_CMAP = 'inferno'
STEM = 'supplement_reproducibility_scatters'


def joint_quantification():
    """log2 protein-group intensity per run, all 8 runs in one DirectLFQ pass."""
    parts, runs = [], {}
    for tag, path in PREPARATIONS:
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


def stats_for(x, y):
    d = (y - x).to_numpy()
    return {'n': len(x),
            'pearson_r': stats.pearsonr(x, y)[0],
            'spearman_rho': stats.spearmanr(x, y)[0],
            'median_log2_difference': float(np.median(d)),
            'iqr_log2_difference': float(np.subtract(*np.percentile(d, [75, 25]))),
            'within_2fold': float(np.mean(np.abs(d) < 1))}


def draw_scatter(ax, x, y, xlabel, ylabel, s, limits):
    """One correlation panel, styled like core.plot_correlation on a shared axis.

    core.plot_correlation builds its own figure and takes no `ax`, so the three
    panels reproduce its content here: 2D-density colouring, y=x diagonal, and
    the r / rho / n box, plus the IQR of the log2 difference — the effect size
    that carries the comparison between panels.
    """
    xv, yv = x.to_numpy(), y.to_numpy()
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
            f"r = {s['pearson_r']:.3f}\n"
            f"ρ = {s['spearman_rho']:.3f}\n"
            f"IQR = {s['iqr_log2_difference']:.2f}\n"
            f"n = {s['n']:,}",
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.85), zorder=6)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    lg, runs = joint_quantification()

    for tag, r in runs.items():
        pairs = [stats.pearsonr(*lg[[a, b]].dropna().to_numpy().T)[0]
                 for a, b in itertools.combinations(r, 2)]
        print(f'{tag}: {lg[r].notna().any(axis=1).sum():,} protein groups, '
              f'single-injection pairwise r = {np.mean(pairs):.4f} '
              f'(range {min(pairs):.4f}-{max(pairs):.4f})')

    complete = lg.dropna()
    print(f'\ncomplete cases across all 8 runs: {len(complete):,} protein groups '
          f'of {len(lg):,}')

    # Panel definitions: (x series, y series, x label, y label, key).
    panels = []
    for tag in ('April', 'June'):
        w = complete[runs[tag]]
        panels.append((w.iloc[:, :2].mean(axis=1), w.iloc[:, 2:].mean(axis=1),
                       f'log₂ {tag}, reps 1-2', f'log₂ {tag}, reps 3-4',
                       f'within {tag}'))
    panels.append((complete[runs['April']].mean(axis=1),
                   complete[runs['June']].mean(axis=1),
                   'log₂ April, all reps', 'log₂ June, all reps',
                   'April vs June'))

    every = np.concatenate([np.concatenate([p[0].to_numpy(), p[1].to_numpy()])
                            for p in panels])
    pad = 0.03 * (every.max() - every.min())
    limits = (every.min() - pad, every.max() + pad)

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
    rows, table = [], []
    for ax, (x, y, xlabel, ylabel, key) in zip(axes, panels):
        s = stats_for(x, y)
        draw_scatter(ax, x, y, xlabel, ylabel, s, limits)
        table.append({'comparison': key, **s})
        rows.append(pd.DataFrame({'comparison': key, 'protein_group': x.index,
                                  'x_log2_intensity': x.to_numpy(),
                                  'y_log2_intensity': y.to_numpy(),
                                  'log2_difference': (y - x).to_numpy()}))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight',
                dpi=RASTER_DPI)
    fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300,
                bbox_inches='tight')

    summary = pd.DataFrame(table).set_index('comparison')
    print('\n' + summary.round(3).to_string())
    across = summary.loc['April vs June', 'iqr_log2_difference']
    within = summary.loc[['within April', 'within June'],
                         'iqr_log2_difference'].mean()
    print(f'\nIQR of the log2 difference: {across:.3f} across preparations vs '
          f'{within:.3f} within one, i.e. {across / within:.1f}x wider')

    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'), index=False)
    summary.reset_index().to_csv(
        os.path.join(OUTDIR, f'{STEM}_statistics.csv'), index=False)
    print(f'\nSaved {STEM} to {OUTDIR}')


if __name__ == '__main__':
    main()
