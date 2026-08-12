"""Supplementary figure 3b — peptides and their reproducibility vs sample volume."""

import os

import numpy as np
import matplotlib.pyplot as plt

import spec_analytics as core
import common_suppl3 as cs
import common_figure2 as cf

ROOT = os.path.join(cs.FIG2_INPUT, 'H032_E306')
STEM = 'panel_e_peptides_volume'

METHODS = ['SPEC', 'PAC', 'ISD+']

# Display names as they appear in the assembled figure, so no relabelling in
# Illustrator is needed: the folder tree says SPEC, the paper says SAX SPEC.
DISPLAY = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}
VOLUMES = {'5uL': 5, '10uL': 10, '40uL': 40, '100uL': 100, '200uL': 200}
LEVELS = list(VOLUMES.values())
COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}

EXCLUDE_REPLICATE = 3

summaries, intensities = {}, {}
for method in METHODS:
    for folder, level in VOLUMES.items():
        path = os.path.join(ROOT, method, folder, f'{folder}.parquet')
        pep = cs.read_peptides(path)
        runs = sorted(pep['Run'].unique())
        if len(runs) != 4:
            raise ValueError(f'{method} {folder}: {len(runs)} runs, expected 4')
        keep = [r for r in runs if r != runs[EXCLUDE_REPLICATE - 1]]
        summaries[(method, level)] = cs.summarise(pep[pep['Run'].isin(keep)])
        intensities[(method, level)] = cs.read_precursor_sums(path, runs=keep)
    print(f'  {method} done')

print('\npeptides per replicate and the CV < 20 % subset:')
cs.report(summaries, 'volume_uL')
cs.report_intensity(intensities, 'volume_uL')

series = [{'label': DISPLAY[m], 'method': m, 'color': COLOR[m],
           'totals': {v: summaries[(m, v)]['per_run'] for v in LEVELS},
           'cv20': {v: summaries[(m, v)]['n_cv20'] for v in LEVELS},
           'intensity': {v: intensities[(m, v)].to_numpy() for v in LEVELS}}
          for m in METHODS]

fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
ax_r = ax.twinx()
heights, points = cs.draw_grouped_overlapping(ax, LEVELS, series)
lines = cs.draw_right_lines(ax_r, LEVELS, series)
cs.style_axes(ax, xlabel='Sample volume [µL]')
cs.style_right_axis(ax, ax_r, max(v.max() for v in intensities.values()))
cf.dual_legends(ax, cf.method_key([(DISPLAY[m], COLOR[m]) for m in METHODS]),
                cs.metric_key())
cs.finish(fig, [ax, ax_r], points, STEM)

cs.write_sourcedata(cs.source_rows(summaries, DISPLAY.get, 'volume_uL',
                                   'volume_uL')
                    + cs.intensity_rows(intensities, DISPLAY.get, 'volume_uL'),
                    STEM)
print(f'Saved {STEM} to {cs.OUTDIR}')
