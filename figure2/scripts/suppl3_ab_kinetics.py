"""Supplementary figure 3a, b — digestion kinetics at the protein-group level (H032_E256)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from scipy import stats
import spec_analytics as core
import common_figure2 as cf
import sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'supplementary_figure3', 'scripts')))
import common_suppl3 as cs
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
FOLDER = {'SAX SPEC': 'SPEC', 'ISD+': 'ISD+'}
REPORTS = {(m, t): os.path.join(INPUT, 'H032_E256', FOLDER[m], f'{t}min',
                                'report.parquet')
           for m in ('SAX SPEC', 'ISD+')
           for t in (5, 10, 15, 30, 60, 90, 120, 150)}
CACHE = _cfg.data_dir(__file__, 'e256_per_run_split.parquet')
PG_CV_CACHE = _cfg.data_dir(__file__, 'e256_pg_cv20.parquet')
OUTDIR = _cfg.output_dir_of('supplementary_figure3')
TIME_MIN = {'A': 5, 'B': 10, 'C': 15, 'D': 30, 'E': 60, 'F': 90, 'G': 120,
            'H': 150}
TIMES = [5, 10, 15, 30, 60, 90, 120, 150]
METHODS = ['SAX SPEC', 'ISD+']
COLOR = {'SAX SPEC': core.PALETTE_SINGLE[0], 'ISD+': core.PALETTE_SINGLE[5]}
N_REPLICATES = 4
QVALUE = 0.01
BATCH_ROWS = 500_000
def method_of(column):
    return 'ISD+' if column <= 4 else 'SAX SPEC'
def per_run_metrics():
    """Per-run counts, MC0 rates and summed quantity; cached, it is a slow pass.
    The report holds all 64 runs at 9.0 M rows, and the missed-cleavage count has
    to touch every distinct stripped sequence, so this is the one place in the
    figure where recomputing is expensive enough to justify a cache. Delete
    `data\\e256_per_run.parquet` to force it.
    """
    if os.path.exists(CACHE):
        print(f'reusing cached per-run metrics: {os.path.basename(CACHE)}')
        return pd.read_parquet(CACHE)
    columns = ['Run', 'Protein.Group', 'Stripped.Sequence',
               'Precursor.Quantity', 'Decoy', 'Q.Value', 'PG.Q.Value']
    mc_memo, acc = {}, {}
    for (method, time_min), path in REPORTS.items():
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=columns):
            d = batch.to_pandas()
            d = d[(d['Decoy'] == 0) & (d['Q.Value'] <= QVALUE)
                  & (d['PG.Q.Value'] <= QVALUE)]
            if d.empty:
                continue
            for seq in d['Stripped.Sequence'].unique():
                if seq not in mc_memo:
                    mc_memo[seq] = core.count_missed_cleavages(
                        seq, protease='trypsin')
            is_mc0 = d['Stripped.Sequence'].map(mc_memo) == 0
            for run, g in d.assign(is_mc0=is_mc0).groupby('Run', sort=False):
                a = acc.setdefault((method, time_min, run),
                                   {'pg': set(), 'n': 0, 'n_mc0': 0,
                                    'q': 0.0, 'q_mc0': 0.0})
                a['pg'].update(g['Protein.Group'].unique())
                a['n'] += len(g)
                a['n_mc0'] += int(g['is_mc0'].sum())
                a['q'] += float(g['Precursor.Quantity'].sum())
                a['q_mc0'] += float(g.loc[g['is_mc0'],
                                          'Precursor.Quantity'].sum())
    rows = []
    for (method, time_min, run), a in acc.items():
        letter, column = run[-2], int(run[-1])
        rows.append({
            'run': run, 'well': f'{letter}{column}',
            'time_min': time_min, 'method': method,
            'replicate': column if column <= 4 else column - 4,
            'protein_groups': len(a['pg']),
            'mc0_rate_by_count': 100 * a['n_mc0'] / a['n'],
            'mc0_rate_by_intensity': 100 * a['q_mc0'] / a['q'],
            'precursor_sum_raw': a['q'],
        })
    out = pd.DataFrame(rows).sort_values(['method', 'time_min', 'replicate'])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out
def pg_cv20_table():
    """Protein groups below 20 % CV per (method, digestion time)."""
    if os.path.exists(PG_CV_CACHE):
        return pd.read_parquet(PG_CV_CACHE)
    rows = []
    for (method, time_min), path in REPORTS.items():
        cv = cf.protein_group_cv(path)
        rows.append({'method': method, 'time_min': time_min,
                     'pg_cv20': cf.n_cv20(cv),
                     'median_pg_cv_pct': float(100 * cv.median())})
        print(f'  PG CV {method} {time_min} min done')
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(PG_CV_CACHE), exist_ok=True)
    out.to_parquet(PG_CV_CACHE, index=False)
    return out
per_run = per_run_metrics()
pg_cv = pg_cv20_table().set_index(['method', 'time_min'])
counts = per_run.groupby(['method', 'time_min']).size()
if not (counts == N_REPLICATES).all():
    raise ValueError(f'expected {N_REPLICATES} replicates per cell: '
                     f'{counts[counts != N_REPLICATES].to_dict()}')
for metric in ['protein_groups', 'mc0_rate_by_count', 'mc0_rate_by_intensity']:
    table = per_run.pivot_table(index='time_min', columns='method',
                               values=metric, aggfunc=['mean', 'std'])
    print(f'\n{metric} (mean, SD over {N_REPLICATES} replicates):')
    print(table.round(1 if metric == 'protein_groups' else 2).to_string())
print('\nSAX SPEC at 5 min vs ISD+ at its best time:')
spec5 = per_run[(per_run['method'] == 'SAX SPEC')
                & (per_run['time_min'] == 5)]['protein_groups'].mean()
isd_best = per_run[per_run['method'] == 'ISD+'].groupby('time_min')[
    'protein_groups'].mean()
print(f'  SPEC 5 min {spec5:,.0f} vs ISD+ best {isd_best.max():,.0f} '
      f'at {isd_best.idxmax()} min  ({spec5 / isd_best.max() - 1:+.1%})')
print('\nstep tests on the count-weighted rate (Welch, 4 + 4 replicates):')
for method in METHODS:
    sub = per_run[per_run['method'] == method]
    line = [f'  {method:9s}']
    for a, b in zip(TIMES[:-1], TIMES[1:]):
        x = sub.loc[sub['time_min'] == a, 'mc0_rate_by_count']
        y = sub.loc[sub['time_min'] == b, 'mc0_rate_by_count']
        p = stats.ttest_ind(x, y, equal_var=False).pvalue
        line.append(f'{a}->{b}: {y.mean() - x.mean():+.2f} (p={p:.3f})')
    print('  '.join(line))
cross = per_run.pivot_table(index='time_min', columns='method',
                            values='mc0_rate_by_count', aggfunc='mean')
print('\ncount-weighted rate, SAX SPEC minus ISD+ [percentage points]:')
print((cross['SAX SPEC'] - cross['ISD+']).round(2).to_string())
def replicates(method, column):
    return {t: per_run.loc[(per_run['method'] == method)
                           & (per_run['time_min'] == t), column].to_numpy()
            for t in TIMES}
def draw_protein_groups(stem, outdir=OUTDIR):
    """Overlapping bars: protein groups per replicate, CV < 20 % subset dark.
    **The twin axis is gone.** It carried the fully-cleaved signal, a different
    quantity in different units, which is now its own panel (`draw_cleaved`) —
    d to g are one construction with the main figure. Light bars unchanged; only
    the dark bar is new.
    """
    series = [{'label': m, 'method': m, 'color': COLOR[m],
               'totals': replicates(m, 'protein_groups'),
               'cv20': {t: int(pg_cv.loc[(m, t), 'pg_cv20']) for t in TIMES}}
              for m in METHODS]
    fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
    heights, bar_points = cs.draw_grouped_overlapping(ax, TIMES, series)
    cs.style_axes(ax, xlabel='Digestion time [min]', ylabel='Protein groups',
                  ymax=per_run['protein_groups'].max())
    cf.dual_legends(ax, cf.method_key([(m, COLOR[m]) for m in METHODS]),
                    cs.metric_key(with_intensity=False,
                                  total_label='Protein groups'))
    cs.finish(fig, ax, bar_points, stem, outdir=outdir)
    cols = ['series', 'metric', 'method', 'time_min', 'run', 'replicate',
            'value']
    pts = pd.concat([
        per_run.assign(series='replicate point', metric=metric,
                       value=per_run[column])[cols]
        for column, metric in (('protein_groups', 'Protein groups'),
                               ('mc0_rate_by_intensity',
                                'Fully cleaved signal [%]'))],
        ignore_index=True)
    bars = pd.DataFrame(
        [{'series': 'light bar (mean)', 'metric': 'Protein groups',
          'method': label, 'time_min': cat, 'run': '', 'replicate': np.nan,
          'value': mean} for label, cat, mean, _dark in heights]
        + [{'series': 'dark bar (condition)',
            'metric': 'Protein groups at CV < 20 %', 'method': label,
            'time_min': cat, 'run': '', 'replicate': np.nan, 'value': dark}
           for label, cat, _mean, dark in heights])
    extra = pd.DataFrame([
        {'series': 'condition summary',
         'metric': 'Median protein-group CV [%]', 'method': m, 'time_min': t,
         'run': '', 'replicate': np.nan,
         'value': round(float(pg_cv.loc[(m, t), 'median_pg_cv_pct']), 2)}
        for m in METHODS for t in TIMES])
    pd.concat([pts, bars[cols], extra[cols]], ignore_index=True).to_csv(
        os.path.join(outdir, f'{stem}_sourcedata.csv'), index=False)
def draw_cleaved(stem, rate_column='mc0_rate_by_intensity', outdir=OUTDIR):
    """Fully cleaved signal against digestion time, as a LINE plot.
    A line, not bars: the quantity is a kinetic curve read for its SHAPE — how
    fast each workflow approaches its plateau and where the two converge — and
    bars make the reader compare eight pairs of heights to see a trend that a
    line states directly. The bars also spent the whole 0-100 axis on a range the
    data never leaves.
    Zero-based still, so the eye is not invited to read a 15-point rise as a
    fourfold one, with the replicates overlaid as points on each marker.
    Signal-weighted, matching every other cleavage number in the paper. The
    count-weighted column is computed too and stays in the source data.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
    for m in METHODS:
        reps = replicates(m, rate_column)
        means = [float(np.mean(reps[t])) for t in TIMES]
        ax.plot(range(len(TIMES)), means, color=COLOR[m], lw=1.8, marker='o',
                ms=5, mec='white', mew=0.8, label=m, zorder=4)
        for i_t, t in enumerate(TIMES):
            v = reps[t]
            ax.scatter(i_t + rng.uniform(-0.06, 0.06, size=v.size), v, s=18,
                       color='black', alpha=0.65, linewidth=0.3,
                       edgecolor='white', zorder=5)
    ax.set_xticks(range(len(TIMES)))
    ax.set_xticklabels([str(t) for t in TIMES], fontsize=cs.FONTSIZE)
    ax.set_xlim(-0.4, len(TIMES) - 0.6)
    ax.set_xlabel('Digestion time [min]', fontsize=cs.FONTSIZE)
    ax.set_ylabel('Fully cleaved signal [%]', fontsize=cs.FONTSIZE)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=cs.FONTSIZE)
    ax.legend(loc='lower right', frameon=False, fontsize=cs.LEGEND_FONTSIZE,
              handlelength=1.4, labelspacing=0.3, borderaxespad=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    cf.set_axes_size_inches(fig, [ax], h_in=cf.AXES_H_IN)
    fig.savefig(os.path.join(outdir, f'{stem}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, f'{stem}.png'), dpi=300,
                bbox_inches='tight')
    print(f'  saved {stem} as a line plot')
    cols = ['series', 'metric', 'method', 'time_min', 'run', 'replicate',
            'value']
    pts = pd.concat([
        per_run.assign(series='replicate point', metric=metric,
                       value=per_run[column])[cols]
        for column, metric in (
            ('mc0_rate_by_intensity', 'Fully cleaved signal [%]'),
            ('mc0_rate_by_count', 'Fully cleaved precursors [%]'))],
        ignore_index=True)
    line = pd.DataFrame([
        {'series': 'line marker (mean)',
         'metric': 'Fully cleaved signal [%]', 'method': m, 'time_min': t,
         'run': '', 'replicate': np.nan,
         'value': round(float(np.mean(replicates(m, rate_column)[t])), 3)}
        for m in METHODS for t in TIMES])
    pd.concat([pts, line[cols]], ignore_index=True).to_csv(
        os.path.join(outdir, f'{stem}_sourcedata.csv'), index=False)
if __name__ == '__main__':
    draw_protein_groups('suppl3_a_kinetics_protein_groups')
    draw_cleaved('suppl3_b_cleaved_kinetics')
    print('Saved the kinetics panels to ' + OUTDIR)
