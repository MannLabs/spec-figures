"""Figure 6f + 6g — ubiquitination (KGG) with SPEC (H341_LTH_3)."""
import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2
import spec_analytics as core
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__)
OUTDIR = _cfg.output_dir(__file__)
CONDITIONS = ['In solution', 'SPEC']
LABEL = {'In solution': 'ISD', 'SPEC': 'SAX SPEC'}
FILES = {'In solution': 'diGly_sol_new.csv', 'SPEC': 'diGly_SPEC_new.csv'}
COLOR = {'In solution': core.PALETTE_SINGLE[1], 'SPEC': core.PALETTE_SINGLE[0]}
SITE_KEY = 'PTM_Collapse_key'
FONTSIZE = 8
WRITE_RETIRED = False
BAR_WIDTH_IN = 0.38
SHOW_VALUE_LABELS = False
POINT_SIZE = core.replicate_point_size(BAR_WIDTH_IN)
GO_TERMS = ['Intracellular Protein Transport',
            'Golgi Vesicle Transport',
            'Formation Of Cytoplasmic Translation Initiation Complex',
            'Mitotic Sister Chromatid Segregation',
            'Regulation Of Mitotic Cell Cycle Phase Transition']
FDR_CUTOFF = 0.05
GO_ROWS = [('Unique SPEC', 'Unique_FDR'),
           ('Shared SPEC\n& in solution', 'Shared_FDR')]
GO_CMAP = LinearSegmentedColormap.from_list(
    'go_fdr', ['#FFFFFF', core.PALETTE_SINGLE[0], '#7A2A1C'])
sites, per_run = {}, []
for condition, fname in FILES.items():
    table = pd.read_csv(os.path.join(INPUT, fname), index_col=0)
    runs = [c for c in table.columns if c.startswith('2025')]
    sites[condition] = set(table[SITE_KEY])
    for replicate, run in enumerate(sorted(runs), start=1):
        per_run.append({'condition': condition, 'replicate': replicate,
                        'run': run, 'kgg_sites': int(table[run].notna().sum())})
    print(f'{condition}: {len(runs)} runs, {len(sites[condition]):,} sites in total')
counts = pd.DataFrame(per_run)
means = counts.groupby('condition')['kgg_sites'].mean().reindex(CONDITIONS)
print('\nKGG sites per run:')
print(counts.pivot(index='replicate', columns='condition',
                   values='kgg_sites')[CONDITIONS].to_string())
print(f'means: {means["In solution"]:,.0f} -> {means["SPEC"]:,.0f}  '
      f'({100 * (means["SPEC"] / means["In solution"] - 1):+.0f}%)')
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(2.2, 2.6))
for i, condition in enumerate(CONDITIONS):
    vals = counts.loc[counts['condition'] == condition, 'kgg_sites'].to_numpy()
    ax.bar(i, vals.mean(), 0.6, color=COLOR[condition], edgecolor='black',
           linewidth=0.5, zorder=2)
    ax.scatter(np.full(len(vals), i) + rng.uniform(-0.09, 0.09, len(vals)), vals,
               s=POINT_SIZE, color='black', alpha=0.75, linewidth=0.3, edgecolor='white',
               zorder=5)
    if SHOW_VALUE_LABELS:
        ax.text(i, vals.max() * 1.03, f'{vals.mean():,.0f}', ha='center',
                va='bottom', fontsize=7.5, fontweight='bold')
ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels([LABEL[c] for c in CONDITIONS], fontsize=FONTSIZE,
                   rotation=45, ha='right')
ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
ax.set_ylim(0, counts['kgg_sites'].max() * 1.18)
ax.set_ylabel('KGG sites identified', fontsize=FONTSIZE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.canvas.draw()
ax_w = ax.get_position().width * fig.get_size_inches()[0]
w_data = BAR_WIDTH_IN * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax_w
for patch in ax.patches:
    centre = patch.get_x() + patch.get_width() / 2
    patch.set_width(w_data)
    patch.set_x(centre - w_data / 2)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'panel_f_kgg_sites.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_f_kgg_sites.png'), dpi=300, bbox_inches='tight')
counts.to_csv(os.path.join(OUTDIR, 'panel_f_kgg_sites_sourcedata.csv'), index=False)
only_sol = len(sites['In solution'] - sites['SPEC'])
shared = len(sites['In solution'] & sites['SPEC'])
only_spec = len(sites['SPEC'] - sites['In solution'])
print(f'\noverlap: in solution only {only_sol:,}, shared {shared:,}, '
      f'SPEC only {only_spec:,}')
fig, ax = plt.subplots(figsize=(2.05, 2.05))
v = venn2(subsets=(only_sol, only_spec, shared),
          set_labels=[LABEL[c] for c in CONDITIONS], ax=ax)
for patch_id, condition in (('10', 'In solution'), ('01', 'SPEC')):
    patch = v.get_patch_by_id(patch_id)
    if patch is not None:
        patch.set_color(COLOR[condition])
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)
overlap = v.get_patch_by_id('11')
if overlap is not None:
    overlap.set_color('#9E6E9E')
    overlap.set_alpha(0.9)
    overlap.set_edgecolor('black')
    overlap.set_linewidth(0.5)
for text in v.set_labels:
    if text is not None:
        text.set_fontsize(FONTSIZE)
for text in v.subset_labels:
    if text is not None:
        text.set_fontsize(7.5)
        text.set_text(f'{int(text.get_text()):,}')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'panel_g_kgg_overlap.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_g_kgg_overlap.png'), dpi=300,
            bbox_inches='tight')
pd.DataFrame([{'region': 'In solution only', 'kgg_sites': only_sol},
              {'region': 'shared', 'kgg_sites': shared},
              {'region': 'SPEC only', 'kgg_sites': only_spec}]).to_csv(
    os.path.join(OUTDIR, 'panel_g_kgg_overlap_sourcedata.csv'), index=False)
go = pd.read_csv(os.path.join(INPUT, 'GO_shared_vs_unique_complete.csv'))
go['short'] = go['Term'].str.replace(r'\s*\(GO:\d+\)$', '', regex=True)
picked = go.set_index('short').reindex(GO_TERMS)
missing = picked[picked['Term'].isna()].index.tolist()
if missing:
    raise ValueError(f'terms absent from the GO table: {missing}')
values = np.array([[-np.log10(picked.loc[t, col]) for t in GO_TERMS]
                   for _label, col in GO_ROWS])
significant = np.array([[picked.loc[t, col] < FDR_CUTOFF for t in GO_TERMS]
                        for _label, col in GO_ROWS])
print('\nGO enrichment, -log10(FDR):')
print(pd.DataFrame(values, index=[l for l, _ in GO_ROWS],
                   columns=[t[:34] for t in GO_TERMS]).round(2).to_string())
vmax = float(np.ceil(values[significant].max()))
fig, ax = plt.subplots(figsize=(4.0, 2.3))
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        if significant[i, j]:
            face = GO_CMAP(values[i, j] / vmax)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                   edgecolor='white', linewidth=1.5, zorder=2))
            ax.text(j, i, f'{values[i, j]:.1f}', ha='center', va='center',
                    fontsize=7, color='white' if values[i, j] > 0.55 * vmax else 'black')
        else:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='#F2F2F2',
                                   edgecolor='white', linewidth=1.5, zorder=2))
            ax.text(j, i, 'NS', ha='center', va='center', fontsize=7,
                    color='#777777')
ax.set_xlim(-0.5, values.shape[1] - 0.5)
ax.set_ylim(values.shape[0] - 0.5, -0.5)
ax.set_xticks(range(values.shape[1]))
ax.set_xticklabels([textwrap.fill(t, 22) for t in GO_TERMS], fontsize=7,
                   rotation=45, ha='right')
ax.set_yticks(range(values.shape[0]))
ax.set_yticklabels([l for l, _ in GO_ROWS], fontsize=FONTSIZE)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
mappable = plt.cm.ScalarMappable(cmap=GO_CMAP,
                                 norm=plt.Normalize(vmin=0, vmax=vmax))
cb = fig.colorbar(mappable, ax=ax, orientation='horizontal', location='top',
                  fraction=0.12, pad=0.06)
cb.set_label('−log₁₀(FDR)', fontsize=7)
cb.ax.tick_params(labelsize=7)
cb.outline.set_visible(False)
cb.solids.set_rasterized(False)
cb.solids.set_edgecolor('face')
fig.tight_layout()
if not WRITE_RETIRED:
    plt.close(fig)
    print(f'\nSaved panels f and g to {OUTDIR} '
          '(GO enrichment retired, WRITE_RETIRED=False)')
    raise SystemExit(0)
fig.savefig(os.path.join(OUTDIR, 'panel_go_enrichment.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_go_enrichment.png'), dpi=300,
            bbox_inches='tight')
pd.DataFrame([{'term': t, 'set': label, 'FDR': picked.loc[t, col],
               'minus_log10_FDR': -np.log10(picked.loc[t, col]),
               'significant': bool(picked.loc[t, col] < FDR_CUTOFF)}
              for label, col in GO_ROWS for t in GO_TERMS]).to_csv(
    os.path.join(OUTDIR, 'panel_go_enrichment_sourcedata.csv'), index=False)
print(f'\nSaved panels e, f and g to {OUTDIR}')
