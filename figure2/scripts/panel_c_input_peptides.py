"""Figure 2c — peptides and their reproducibility vs protein input (H032_E305)."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import spec_analytics as core
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..',
                                                'supplementary_figure3',
                                                'scripts')))
OUTDIR = _cfg.output_dir(__file__)
import common_suppl3 as cs
import common_figure2 as cf
ROOT = os.path.join(cs.FIG2_INPUT, 'H032_E305')
STEM = 'panel_c_input_peptides'
METHODS = ['SPEC', 'PAC', 'ISD+']
DISPLAY = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}
INPUT_MAP = {'5ng': 5, '20ng': 20, '50ng': 50, '200ng': 200, '500ng': 500}
LEVELS = sorted(INPUT_MAP.values())
COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}
summaries, intensities = {}, {}
for method in METHODS:
    for folder, level in INPUT_MAP.items():
        path = os.path.join(ROOT, method, folder, 'report.parquet')
        if not os.path.exists(path):
            path = os.path.join(ROOT, method, folder, f'{folder}.parquet')
        s = cs.summarise(cs.read_peptides(path))
        if len(s['runs']) != 4:
            raise ValueError(f'{method} {folder}: {len(s["runs"])} runs, expected 4')
        summaries[(method, level)] = s
    print(f'  {method} done')
print('\npeptides per replicate and the CV < 20 % subset:')
cs.report(summaries, 'input_ng')
series = [{'label': DISPLAY[m], 'method': m, 'color': COLOR[m],
           'totals': {v: summaries[(m, v)]['per_run'] for v in LEVELS},
           'cv20': {v: summaries[(m, v)]['n_cv20'] for v in LEVELS}}
          for m in METHODS]
fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
heights, points = cs.draw_grouped_overlapping(ax, LEVELS, series)
ymax = max(float(np.max(s['totals'][c]))
           for s in series for c in LEVELS)
cs.style_axes(ax, ymax=ymax, xlabel='Protein input amount [ng]')
cf.dual_legends(ax, cf.method_key([(DISPLAY[m], COLOR[m]) for m in METHODS]),
                cs.metric_key(with_intensity=False))
cs.finish(fig, ax, points, STEM, outdir=OUTDIR)
cs.write_sourcedata(cs.source_rows(summaries, DISPLAY.get, 'input_ng',
                                   'input_ng'),
                    STEM, outdir=OUTDIR)
print(f'Saved {STEM} to {OUTDIR}')
