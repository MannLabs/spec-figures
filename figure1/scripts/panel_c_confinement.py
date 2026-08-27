"""Figure 1c — the axial confinement of the bed, as a volume."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spec_analytics as core
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
OUTDIR = _cfg.output_dir(__file__)
NPZ = _cfg.input_dir(__file__, 'confinement_profile.npz')
STEM = 'panel_c_confinement'
AXES_H_IN = 2.50
AXES_W_IN = 2.60
FONTSIZE = 10
LEGEND_FONTSIZE = 10
BED_DIAMETER_UM = 1000.0
NL_PER_UM = np.pi * (BED_DIAMETER_UM / 2) ** 2 / 1e9 * 1e3
CHANNEL_COLOR = {'protein': '#D62728', 'protease': '#1F9FD0'}
z = np.load(NPZ)
depth_um = z['depth']
fig, ax = plt.subplots(figsize=(3.4, 4))
rows = []
for name, colour in CHANNEL_COLOR.items():
    med = z[f'{name}_flat_med_trace']
    q1, q3 = z[f'{name}_flat_q1'], z[f'{name}_flat_q3']
    fwhm_nl = float(z[f'{name}_flat_med']) * NL_PER_UM
    iqr_nl = np.asarray(z[f'{name}_flat_iqr']) * NL_PER_UM
    crest = float(z[f'{name}_draw_depth_um'])
    volume_nl = (depth_um + crest) * NL_PER_UM
    ax.fill_between(volume_nl, q1, q3, color=colour, alpha=0.15, linewidth=0)
    ax.plot(volume_nl, med, color=colour, linewidth=1.6,
            label=f'{name}  {fwhm_nl:.0f} nL  ({iqr_nl[0]:.0f}–{iqr_nl[1]:.0f})')
    ax.plot([crest * NL_PER_UM], [np.nanmax(med)], marker='v', ms=5,
            color=colour, mec='white', mew=0.7, clip_on=False, zorder=6)
    for v, m, a, b in zip(volume_nl, med, q1, q3):
        rows.append({'series': 'median trace', 'channel': name,
                     'volume_from_bed_top_nL': round(float(v), 3),
                     'norm_fluorescence': round(float(m), 5),
                     'q1': round(float(a), 5), 'q3': round(float(b), 5)})
    rows.append({'series': 'summary', 'channel': name,
                 'volume_from_bed_top_nL': '', 'norm_fluorescence': '',
                 'q1': '', 'q3': '',
                 'fwhm_nL_median': round(fwhm_nl, 1),
                 'fwhm_nL_q1': round(float(iqr_nl[0]), 1),
                 'fwhm_nL_q3': round(float(iqr_nl[1]), 1),
                 'fwhm_um_median': round(float(z[f'{name}_flat_med']), 1),
                 'crest_depth_um_median': round(float(z[f'{name}_crest_depth_med']), 1),
                 'drawn_at_depth_um': round(crest, 1),
                 'n_near_flat_columns': int(z[f'{name}_n_flat']),
                 'n_columns_in_trace': int(z[f'{name}_n_crest']),
                 'band_tilt_median_deg': round(float(z[f'{name}_tilt_med']), 1),
                 'paired_crest_offset_um': round(float(z['paired_offset_um']), 1)})
ax.axhline(0.5, color='0.6', linewidth=0.6, linestyle=':')
ax.set_xlabel('Bed volume from the top [nL]', fontsize=FONTSIZE)
ax.set_ylabel('Normalised fluorescence', fontsize=FONTSIZE)
ax.set_xlim(0, 700 * NL_PER_UM)
ax.set_ylim(-0.05, 1.42)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.legend(frameon=False, loc='upper right', fontsize=LEGEND_FONTSIZE,
          handlelength=1.1, labelspacing=0.3)
fig.tight_layout()
core.set_axes_size_inches(fig, ax, w_in=AXES_W_IN, h_in=AXES_H_IN)
print(f'  paired crest offset {float(z["paired_offset_um"]):+.0f} um '
      f'({float(z["paired_offset_um"]) * NL_PER_UM:+.0f} nL)')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, f'{STEM}_sourcedata.csv'),
                          index=False)
for name in CHANNEL_COLOR:
    print(f'  {name:9s} FWHM {float(z[f"{name}_flat_med"]) * NL_PER_UM:5.0f} nL '
          f'from {int(z[f"{name}_n_flat"])} columns of the flat window '
          f'(band tilt median {float(z[f"{name}_tilt_med"]):.0f} deg)')
print(f'Saved {STEM} to {OUTDIR}')
