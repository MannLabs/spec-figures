"""Supplementary figure 3c — peptides and their reproducibility vs protein input."""

import os

import numpy as np
import matplotlib.pyplot as plt

import spec_analytics as core
import common_suppl3 as cs
import common_figure2 as cf

ROOT = os.path.join(cs.FIG2_INPUT, 'H032_E305')
STEM = 'panel_f_peptides_input'

METHODS = ['SPEC', 'PAC', 'ISD+']
DISPLAY = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}
INPUT_MAP = {'5ng': 5, '20ng': 20, '50ng': 50, '200ng': 200, '500ng': 500}
LEVELS = sorted(INPUT_MAP.values())
COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}

summaries, intensities = {}, {}
for method in METHODS:
    for folder, level in INPUT_MAP.items():
        # E305 writes `report.parquet` in most subfolders and `<folder>.parquet`
        # in the rest; figure 2c handles the same split.
        path = os.path.join(ROOT, method, folder, 'report.parquet')
        if not os.path.exists(path):
            path = os.path.join(ROOT, method, folder, f'{folder}.parquet')
        s = cs.summarise(cs.read_peptides(path))
        if len(s['runs']) != 4:
            raise ValueError(f'{method} {folder}: {len(s["runs"])} runs, expected 4')
        summaries[(method, level)] = s
        intensities[(method, level)] = cs.read_precursor_sums(path)
    print(f'  {method} done')

print('\npeptides per replicate and the CV < 20 % subset:')
cs.report(summaries, 'input_ng')
cs.report_intensity(intensities, 'input_ng')

series = [{'label': DISPLAY[m], 'method': m, 'color': COLOR[m],
           'totals': {v: summaries[(m, v)]['per_run'] for v in LEVELS},
           'cv20': {v: summaries[(m, v)]['n_cv20'] for v in LEVELS},
           'intensity': {v: intensities[(m, v)].to_numpy() for v in LEVELS}}
          for m in METHODS]

fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
ax_r = ax.twinx()
heights, points = cs.draw_grouped_overlapping(ax, LEVELS, series)
lines = cs.draw_right_lines(ax_r, LEVELS, series)
cs.style_axes(ax, xlabel='Protein input amount [ng]')
cs.style_right_axis(ax, ax_r, max(v.max() for v in intensities.values()))
cf.dual_legends(ax, cf.method_key([(DISPLAY[m], COLOR[m]) for m in METHODS]),
                cs.metric_key())
cs.finish(fig, [ax, ax_r], points, STEM)

cs.write_sourcedata(cs.source_rows(summaries, DISPLAY.get, 'input_ng',
                                   'input_ng')
                    + cs.intensity_rows(intensities, DISPLAY.get, 'input_ng'),
                    STEM)
print(f'Saved {STEM} to {cs.OUTDIR}')
