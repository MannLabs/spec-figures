"""Figure 2d — peptides and their reproducibility vs detergent load (H032_E307)."""
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
BASE = os.path.join(cs.FIG2_INPUT, 'H032_E307', 'SAX')
STEM = 'panel_d_detergent_peptides'
CONDITIONS = [
    ('no_detergent',   'no detergent'),
    ('2percent_SDC',   '2% SDC'),
    ('4percent_SDC',   '4% SDC'),
    ('0p5percent_SDS', '0.5% SDS'),
    ('2percent_SDS',   '2% SDS'),
    ('10percent_SDS',  '10% SDS'),
]
ORDER = [label for _f, label in CONDITIONS]
TICK = {'no detergent': 'none', '2% SDC': '2%', '4% SDC': '4%',
        '0.5% SDS': '0.5%', '2% SDS': '2%', '10% SDS': '10%'}
CLASS_GROUPS = [('no detergent', ['no detergent']),
                ('SDC', ['2% SDC', '4% SDC']),
                ('SDS', ['0.5% SDS', '2% SDS', '10% SDS'])]
SAX_COLOR = core.PALETTE_SINGLE[0]
summaries, intensities = {}, {}
for folder, label in CONDITIONS:
    path = os.path.join(BASE, folder, f'{folder}.parquet')
    s = cs.summarise(cs.read_peptides(path))
    if len(s['runs']) != 4:
        raise ValueError(f'{label}: {len(s["runs"])} runs, expected 4')
    summaries[('SAX SPEC', label)] = s
print('\npeptides per replicate and the CV < 20 % subset:')
cs.report(summaries, 'condition')
series = [{'label': 'SAX SPEC', 'method': 'SAX SPEC', 'color': SAX_COLOR,
           'totals': {c: summaries[('SAX SPEC', c)]['per_run'] for c in ORDER},
           'cv20': {c: summaries[('SAX SPEC', c)]['n_cv20'] for c in ORDER}}]
fig, ax = plt.subplots(figsize=(cf.PANEL_W_IN, 4))
heights, points = cs.draw_grouped_overlapping(ax, ORDER, series,
                                              class_groups=CLASS_GROUPS)
ymax = max(float(np.max(s['totals'][c]))
           for s in series for c in ORDER)
cs.style_axes(ax, ymax=ymax, xlabel='')
ax.set_xticklabels([TICK[c] for c in ORDER], fontsize=cf.FONTSIZE)
cf.dual_legends(ax, [], cs.metric_key(with_intensity=False))
cs.finish(fig, ax, points, STEM, outdir=OUTDIR)
cs.write_sourcedata(cs.source_rows(summaries, lambda m: m, 'condition',
                                   'condition'),
                    STEM, outdir=OUTDIR)
print(f'Saved {STEM} to {OUTDIR}')
