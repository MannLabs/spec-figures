"""Supplementary figure 3d — peptides and their reproducibility vs detergent load."""

import os

import numpy as np
import matplotlib.pyplot as plt

import spec_analytics as core
import common_suppl3 as cs
import common_figure2 as cf

BASE = os.path.join(cs.FIG2_INPUT, 'H032_E307', 'SAX')
STEM = 'panel_g_peptides_detergent'

# (folder, display label) in plotting order — identical to figure 2d.
CONDITIONS = [
    ('10percent_SDS',  '10% SDS'),
    ('2percent_SDS',   '2% SDS'),
    ('0p5percent_SDS', '0.5% SDS'),
    ('4percent_SDC',   '4% SDC'),
    ('2percent_SDC',   '2% SDC'),
    ('no_detergent',   'no detergent'),
]
ORDER = [label for _f, label in CONDITIONS]
SAX_COLOR = core.PALETTE_SINGLE[0]

summaries, intensities = {}, {}
for folder, label in CONDITIONS:
    path = os.path.join(BASE, folder, f'{folder}.parquet')
    s = cs.summarise(cs.read_peptides(path))
    if len(s['runs']) != 4:
        raise ValueError(f'{label}: {len(s["runs"])} runs, expected 4')
    summaries[('SAX SPEC', label)] = s
    intensities[('SAX SPEC', label)] = cs.read_precursor_sums(path)

print('\npeptides per replicate and the CV < 20 % subset:')
cs.report(summaries, 'condition')
cs.report_intensity(intensities, 'condition')

series = [{'label': 'SAX SPEC', 'method': 'SAX SPEC', 'color': SAX_COLOR,
           'totals': {c: summaries[('SAX SPEC', c)]['per_run'] for c in ORDER},
           'cv20': {c: summaries[('SAX SPEC', c)]['n_cv20'] for c in ORDER},
           'intensity': {c: intensities[('SAX SPEC', c)].to_numpy()
                         for c in ORDER}}]

fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
ax_r = ax.twinx()
heights, points = cs.draw_grouped_overlapping(ax, ORDER, series)
lines = cs.draw_right_lines(ax_r, ORDER, series)
cs.style_axes(ax, xlabel='')
cs.style_right_axis(ax, ax_r, max(v.max() for v in intensities.values()))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
# No method key: one method, so hue carries no information here.
cf.dual_legends(ax, [], cs.metric_key())
cs.finish(fig, [ax, ax_r], points, STEM)

cs.write_sourcedata(cs.source_rows(summaries, lambda m: m, 'condition',
                                   'condition')
                    + cs.intensity_rows(intensities, lambda m: m, 'condition'),
                    STEM)
print(f'Saved {STEM} to {cs.OUTDIR}')
