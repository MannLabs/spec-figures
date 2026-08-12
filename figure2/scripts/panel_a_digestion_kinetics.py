"""Figure 2, layout panel a — digestion kinetics (H032_E256)."""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from scipy import stats

import spec_analytics as core
import common_figure2 as cf

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
OUTDIR = _cfg.output_dir(__file__)

# Plate map: row letter = digestion time, column = workflow and replicate.
# Columns 1-4 are the SDB-RPS in-solution digest (ISD+), 5-8 are SAX SPEC.
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
    # The memo for missed cleavages is shared across all sixteen searches, which is
    # what keeps this affordable: the same stripped sequences recur throughout.
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


per_run = per_run_metrics()
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


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def draw_panel(rate_column, right_label, legend_label, stem, outdir=OUTDIR):
    def replicates(method, column):
        return {t: per_run.loc[(per_run['method'] == method)
                               & (per_run['time_min'] == t), column].to_numpy()
                for t in TIMES}

    series = [{'label': f'{m} — protein groups', 'method': m,
               'column': 'protein_groups', 'color': COLOR[m], 'axis': 'left',
               'values': replicates(m, 'protein_groups')} for m in METHODS]
    series += [{'label': f'{m} — {right_label}', 'method': m,
                'column': rate_column, 'color': COLOR[m], 'axis': 'right',
                'values': replicates(m, rate_column)} for m in METHODS]

    fig, ax_l = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
    ax_r = ax_l.twinx()
    heights, bar_points = cf.draw_grouped_dual(ax_l, ax_r, TIMES, series)
    cf.style_dual_axes(
        ax_l, ax_r, left_label='Protein groups', right_label=right_label,
        left_max=per_run['protein_groups'].max(),
        right_max=per_run[rate_column].max())
    ax_l.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax_l.set_xlabel('Digestion time [min]', fontsize=cf.FONTSIZE)
    cf.dual_legends(ax_l,
                    cf.method_key([(m, COLOR[m]) for m in METHODS]),
                    cf.metric_key('Protein groups', legend_label))

    fig.tight_layout()
    cf.set_axes_size_inches(fig, [ax_l, ax_r], h_in=cf.AXES_H_IN)
    cf.finish_points(ax_l, bar_points)
    print(f'{stem}: bar width '
          f'{cf.bar_width_inches(fig, ax_l, len(METHODS), len(TIMES)):.3f} in')
    fig.savefig(os.path.join(outdir, f'{stem}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(outdir, f'{stem}.png'), dpi=300,
                bbox_inches='tight')

    cols = ['series', 'metric', 'method', 'time_min', 'run', 'replicate',
            'value']
    label_of = {s['label']: s for s in series}
    pts = pd.concat([
        per_run.assign(series='replicate point', metric=metric,
                       value=per_run[column])[cols]
        for column, metric in (('protein_groups', 'Protein groups'),
                               (rate_column, right_label))],
        ignore_index=True)
    bars = pd.DataFrame([
        {'series': 'bar height (mean)',
         'metric': ('Protein groups' if label_of[label]['axis'] == 'left'
                    else right_label),
         'method': label_of[label]['method'], 'time_min': cat, 'run': '',
         'replicate': np.nan, 'value': mean}
        for label, cat, mean in heights])
    pd.concat([pts, bars[cols]], ignore_index=True).to_csv(
        os.path.join(outdir, f'{stem}_sourcedata.csv'), index=False)


if __name__ == '__main__':
    draw_panel('mc0_rate_by_count', 'Fully cleaved precursors [%]',
               'Fully cleaved', 'panel_a_digestion_kinetics')
    print(f'\nSaved the kinetics panel to {OUTDIR}')
