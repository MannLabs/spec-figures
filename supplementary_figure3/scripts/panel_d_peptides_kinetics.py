"""Supplementary figure 3a — peptides and their reproducibility vs digestion time."""

import os

import numpy as np
import matplotlib.pyplot as plt

import spec_analytics as core
import common_suppl3 as cs
import common_figure2 as cf

# One search per (workflow, digestion time), matching figure 2a.
FOLDER = {'SAX SPEC': 'SPEC', 'ISD+': 'ISD+'}
REPORTS = {(m, t): os.path.join(cs.FIG2_INPUT, 'H032_E256', FOLDER[m],
                                f'{t}min', 'report.parquet')
           for m in ('SAX SPEC', 'ISD+')
           for t in (5, 10, 15, 30, 60, 90, 120, 150)}
CACHE = cs.cache_path('e256_peptide_cv_split.parquet')
STEM = 'panel_d_peptides_kinetics'

# Same plate map as figure 2a: row letter = digestion time, columns 1-4 are the
# SDB-RPS in-solution digest (ISD+), 5-8 are SAX SPEC.
TIME_MIN = {'A': 5, 'B': 10, 'C': 15, 'D': 30, 'E': 60, 'F': 90, 'G': 120,
            'H': 150}
TIMES = [5, 10, 15, 30, 60, 90, 120, 150]
METHODS = ['SAX SPEC', 'ISD+']
COLOR = {'SAX SPEC': core.PALETTE_SINGLE[0], 'ISD+': core.PALETTE_SINGLE[5]}


def method_of(column):
    return 'ISD+' if column <= 4 else 'SAX SPEC'


# ---------------------------------------------------------------------------
# Per-condition peptide counts and CVs
# ---------------------------------------------------------------------------
def build():
    import pandas as pd
    if os.path.exists(CACHE):
        print(f'reusing cache {os.path.basename(CACHE)}')
        return pd.read_parquet(CACHE)

    print(f'reading {len(REPORTS)} E256 searches (filtered at read time)...')
    pep = pd.concat([cs.read_peptides(path).assign(method=m, time_min=t)
                     for (m, t), path in REPORTS.items()], ignore_index=True)

    sums = {(m, t): cs.read_precursor_sums(path)
            for (m, t), path in REPORTS.items()}

    rows = []
    for (method, t), g in pep.groupby(['method', 'time_min'], sort=False):
        # CVs are computed WITHIN each condition's 4 replicates, never across the
        # series — a CV pooled over digestion times would measure the time effect.
        s = cs.summarise(g)
        if len(s['runs']) != 4:
            raise ValueError(f'{method} {t} min: {len(s["runs"])} runs, expected 4')
        intensity = sums[(method, int(t))].reindex(s['runs'])
        if intensity.isna().any():
            raise ValueError(f'{method} {t} min: intensity missing for '
                             f'{list(intensity[intensity.isna()].index)}')
        rows.append({'method': method, 'time_min': int(t),
                     'per_run': s['per_run'], 'runs': s['runs'],
                     'intensity': intensity.to_numpy(),
                     'union': s['union'], 'n_with_cv': s['n_with_cv'],
                     'n_cv20': s['n_cv20'],
                     'median_cv_pct': s['median_cv_pct']})
    out = pd.DataFrame(rows).sort_values(['method', 'time_min'])
    out.to_parquet(CACHE, index=False)
    return out


table = build()
summaries = {(r['method'], r['time_min']):
             {'per_run': list(r['per_run']), 'runs': list(r['runs']),
              'union': r['union'], 'n_with_cv': r['n_with_cv'],
              'n_cv20': r['n_cv20'], 'median_cv_pct': r['median_cv_pct']}
             for _, r in table.iterrows()}
import pandas as pd                                            # noqa: E402
intensities = {(r['method'], r['time_min']):
               pd.Series(list(r['intensity']), index=list(r['runs']))
               for _, r in table.iterrows()}

print('\npeptides per replicate and the CV < 20 % subset:')
cs.report(summaries, 'time_min')
cs.report_intensity(intensities, 'time_min')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
series = [{'label': m, 'method': m, 'color': COLOR[m],
           'totals': {t: summaries[(m, t)]['per_run'] for t in TIMES},
           'cv20': {t: summaries[(m, t)]['n_cv20'] for t in TIMES},
           'intensity': {t: intensities[(m, t)].to_numpy() for t in TIMES}}
          for m in METHODS]

fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
ax_r = ax.twinx()
heights, points = cs.draw_grouped_overlapping(ax, TIMES, series)
lines = cs.draw_right_lines(ax_r, TIMES, series)
cs.style_axes(ax, xlabel='Digestion time [min]')
cs.style_right_axis(ax, ax_r, max(v.max() for v in intensities.values()))
cf.dual_legends(ax, cf.method_key([(m, COLOR[m]) for m in METHODS]),
                cs.metric_key())
cs.finish(fig, [ax, ax_r], points, STEM)

cs.write_sourcedata(cs.source_rows(summaries, lambda m: m, 'time_min',
                                   'time_min')
                    + cs.intensity_rows(intensities, lambda m: m, 'time_min'),
                    STEM)
print(f'Saved {STEM} to {cs.OUTDIR}')
