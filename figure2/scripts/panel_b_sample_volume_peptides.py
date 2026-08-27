"""Figure 2b — peptides and their reproducibility vs sample volume (H032_E306)."""
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
import common_suppl3 as cs
import common_figure2 as cf
OUTDIR = _cfg.output_dir(__file__)
ROOT = os.path.join(cf.INPUT if hasattr(cf, 'INPUT') else
                    _cfg.input_dir(__file__), 'H032_E306')
STEM = 'panel_b_sample_volume_peptides'
METHODS = ['SPEC', 'PAC', 'ISD+']
DISPLAY = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}
VOLUMES = {'5uL': 5, '10uL': 10, '40uL': 40, '100uL': 100, '200uL': 200}
LEVELS = list(VOLUMES.values())
COLOR = {'SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}
EXCLUDE_REPLICATE = 3
summaries = {}
for method in METHODS:
    for folder, level in VOLUMES.items():
        path = os.path.join(ROOT, method, folder, f'{folder}.parquet')
        pep = cs.read_peptides(path)
        runs = sorted(pep['Run'].unique())
        if len(runs) != 4:
            raise ValueError(f'{method} {folder}: {len(runs)} runs, expected 4')
        keep = [r for r in runs if r != runs[EXCLUDE_REPLICATE - 1]]
        summaries[(method, level)] = cs.summarise(pep[pep['Run'].isin(keep)])
    print(f'  {method} done')
print('\npeptides per replicate and the CV < 20 % subset:')
cs.report(summaries, 'volume_uL')
series = [{'label': DISPLAY[m], 'method': m, 'color': COLOR[m],
           'totals': {v: summaries[(m, v)]['per_run'] for v in LEVELS},
           'cv20': {v: summaries[(m, v)]['n_cv20'] for v in LEVELS}}
          for m in METHODS]
fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
heights, points = cs.draw_grouped_overlapping(ax, LEVELS, series)
ymax = max(float(np.max(s['totals'][v])) for s in series for v in LEVELS)
cs.style_axes(ax, xlabel='Sample volume [µL]', ymax=ymax)
cf.dual_legends(ax, cf.method_key([(DISPLAY[m], COLOR[m]) for m in METHODS]),
                cs.metric_key(with_intensity=False))
cs.finish(fig, ax, points, STEM, outdir=OUTDIR)
cs.write_sourcedata(cs.source_rows(summaries, DISPLAY.get, 'volume_uL',
                                   'volume_uL'), STEM, outdir=OUTDIR)
print(f'Saved {STEM} to {OUTDIR}')
