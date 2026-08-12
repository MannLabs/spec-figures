"""Figure 6h — pathway coverage of the ubiquitinome, SAX-SPEC vs in-solution."""

import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spec_analytics as core

core.init_plotting()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__, 'diGly')
PROFILES = os.path.join(INPUT, 'collapsed_site_profiles.csv')
GMT = os.path.join(INPUT, 'gprofiler_hsapiens.gmt')
OUTDIR = _cfg.output_dir(__file__)

SOURCE_PREFIXES = ('GO:', 'REAC:', 'CORUM:')
MIN_TERM_SIZE, MAX_TERM_SIZE = 10, 700

# The five processes the manuscript already highlights, plus the ubiquitin-
# proteasome machinery as a positive control both workflows should see.
HIGHLIGHT_TERMS = [
    'GO:0006886',         # intracellular protein transport
    'GO:0048193',         # Golgi vesicle transport
    'REAC:R-HSA-72649',   # translation initiation complex formation
    'GO:1901990',         # regulation of mitotic cell cycle phase transition
    'GO:0000070',         # mitotic sister chromatid segregation
    'REAC:R-HSA-983168',  # antigen processing: ubiquitination & proteasome
]

COLOR = {'SAX SPEC': core.PALETTE_SINGLE[0],
         'ISD': core.PALETTE_SINGLE[1]}
SERIES = ['SAX SPEC', 'ISD']

FONTSIZE = 8
BAR_IN = 0.25
GROUP_PITCH_IN = 0.62
AXES_W_IN = 2.45
LABEL_WRAP = 34
# Scatter panel: square, since it is an x-against-y comparison of the same
# quantity, and rasterised because 10k points would crawl in Illustrator.
SCATTER_AXES_IN = 2.50
RASTER_DPI = 600


def wrap(name, n_members):
    """Pathway name wrapped over as many lines as it needs, plus its size.

    Wrapped rather than truncated: the exploratory version cut names at 46
    characters, which turned "regulation of mitotic cell cycle phase transition"
    into "...phase transit" mid-word.
    """
    return '\n'.join(textwrap.wrap(name, LABEL_WRAP) + [f'({n_members} proteins)'])


def detected_genes():
    """Gene groups with >= 1 detected KGG site, per workflow."""
    profiles = pd.read_csv(PROFILES, low_memory=False)
    out = {}
    for label, prefix in (('SAX SPEC', 'SPEC_'), ('ISD', 'ISD_')):
        columns = [c for c in profiles.columns if c.startswith(prefix)]
        if not columns:
            raise ValueError(f'no {prefix}* replicate columns in {PROFILES}')
        subset = profiles[profiles[columns].notna().any(axis=1)]
        out[label] = set(subset['Gene_group'].dropna())
        print(f'{label:12s} {len(columns)} replicates, '
              f'{len(subset):,} sites, {len(out[label]):,} gene groups')
    return out


def coverage_table(genes):
    """One row per pathway: size and the fraction covered by each workflow."""
    rows = []
    with open(GMT, encoding='utf-8') as handle:
        for line in handle:
            fields = line.rstrip('\n').split('\t')
            term_id, term_name = fields[0], fields[1]
            members = set(fields[2:])
            if not term_id.startswith(SOURCE_PREFIXES):
                continue
            if not MIN_TERM_SIZE <= len(members) <= MAX_TERM_SIZE:
                continue
            rows.append({
                'term_id': term_id, 'term_name': term_name,
                'n_members': len(members),
                'SAX SPEC': 100 * len(members & genes['SAX SPEC']) / len(members),
                'ISD': 100 * len(members & genes['ISD']) / len(members),
            })
    table = pd.DataFrame(rows)
    table['delta'] = table['SAX SPEC'] - table['ISD']
    return table


genes = detected_genes()
coverage = coverage_table(genes)

n_higher = int((coverage['delta'] > 0).sum())
n_lower = int((coverage['delta'] < 0).sum())
pct_higher = 100 * n_higher / len(coverage)
med_spec = coverage['SAX SPEC'].median()
med_isd = coverage['ISD'].median()
print(f'\n{len(coverage):,} pathways ({"/".join(p.rstrip(":") for p in SOURCE_PREFIXES)}, '
      f'{MIN_TERM_SIZE}-{MAX_TERM_SIZE} genes)')
print(f'median coverage: SPEC {med_spec:.1f}% vs in-solution {med_isd:.1f}%')
print(f'SPEC higher in {n_higher:,} ({pct_higher:.1f}%), '
      f'in-solution higher in {n_lower:,}, equal in '
      f'{len(coverage) - n_higher - n_lower:,}')

highlight = (coverage.set_index('term_id')
             .reindex(HIGHLIGHT_TERMS).reset_index())
if highlight['term_name'].isna().any():
    missing = highlight.loc[highlight['term_name'].isna(), 'term_id'].tolist()
    raise ValueError(f'highlighted terms absent from the GMT or size-filtered: {missing}')
print(f'\n{"pathway":58s} {"genes":>5s} {"SPEC%":>7s} {"ISD%":>7s}')
for _, r in highlight.iterrows():
    print(f'{r.term_name[:58]:58s} {r.n_members:5d} {r["SAX SPEC"]:6.1f}% '
          f'{r["ISD"]:6.1f}%')

# ---------------------------------------------------------------------------
# Panel h — the six highlighted pathways
# ---------------------------------------------------------------------------
order = highlight.iloc[::-1].reset_index(drop=True)   # first term at the top
y = np.arange(len(order))
offset = {'SAX SPEC': 0.5 * BAR_IN / GROUP_PITCH_IN,
          'ISD': -0.5 * BAR_IN / GROUP_PITCH_IN}
height = BAR_IN / GROUP_PITCH_IN

fig, ax = plt.subplots(figsize=(6.0, 3.2))
for label in SERIES:
    ax.barh(y + offset[label], order[label], height=height, color=COLOR[label],
            edgecolor='black', linewidth=0.5, label=label, zorder=2)
    for i, value in enumerate(order[label]):
        ax.text(value + 1.2, y[i] + offset[label], f'{value:.0f}%',
                va='center', ha='left', fontsize=FONTSIZE)

ax.set_yticks(y)
ax.set_yticklabels([wrap(n, m) for n, m in zip(order['term_name'],
                                               order['n_members'])],
                   fontsize=FONTSIZE)
ax.set_ylim(-0.5, len(order) - 0.5)
ax.set_xlim(0, 100)
# Two lines: on one it is ~3.5 in of text centred on a 2.45 in axes, so it
# overhangs the panel on both sides and the tight bbox clips its right end.
ax.set_xlabel('% of pathway proteins with\n$\\geq$1 ubiquitination site detected',
              fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

from matplotlib.lines import Line2D                                    # noqa: E402

handles, labels = ax.get_legend_handles_labels()
for label in SERIES:
    median = coverage[label].median()
    ax.axvline(median, color='white', linewidth=1.7, zorder=5)
    ax.axvline(median, color=COLOR[label], linewidth=1.0,
               linestyle=(0, (4, 2)), zorder=6)
    print(f'  median of all pathways, {label}: {median:.1f}%')

handles.append(Line2D([0], [0], color='#595959', linewidth=1.0,
                      linestyle=(0, (4, 2))))
labels.append('median, all pathways')

ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3,
          frameon=False, fontsize=FONTSIZE, handlelength=1.2, handleheight=0.9,
          handletextpad=0.4, borderpad=0.0, columnspacing=1.0)

fig.tight_layout()
core.set_axes_size_inches(fig, ax, w_in=AXES_W_IN,
                          h_in=len(order) * GROUP_PITCH_IN)
fig.savefig(os.path.join(OUTDIR, 'panel_h_pathway_coverage.pdf'),
            bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'panel_h_pathway_coverage.png'), dpi=300,
            bbox_inches='tight')

highlight.assign(series='bar length')[
    ['series', 'term_id', 'term_name', 'n_members', 'SAX SPEC', 'ISD',
     'delta']].to_csv(
    os.path.join(OUTDIR, 'panel_h_pathway_coverage_sourcedata.csv'), index=False)

# ---------------------------------------------------------------------------
# Supplement — every pathway
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 3.4))
limit = max(coverage['SAX SPEC'].max(), coverage['ISD'].max()) * 1.02
cloud = ax.scatter(coverage['ISD'], coverage['SAX SPEC'], s=4,
                   alpha=0.18, color=COLOR['SAX SPEC'], edgecolor='none')
cloud.set_rasterized(True)
ax.plot([0, limit], [0, limit], color='black', linestyle='--', linewidth=0.7,
        zorder=3)
ax.text(0.04, 0.96,
        f'n = {len(coverage):,} pathways\nSPEC higher in {pct_higher:.0f}%\n'
        f'median {med_spec:.0f}% vs {med_isd:.0f}%',
        transform=ax.transAxes, fontsize=FONTSIZE, va='top', ha='left')
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_aspect('equal')
ax.set_xlabel('ISD coverage [%]', fontsize=FONTSIZE)
ax.set_ylabel('SAX SPEC coverage [%]', fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
core.set_axes_size_inches(fig, ax, w_in=SCATTER_AXES_IN, h_in=SCATTER_AXES_IN)
fig.savefig(os.path.join(OUTDIR, 'supplement_pathway_coverage_scatter.pdf'),
            bbox_inches='tight', dpi=RASTER_DPI)
fig.savefig(os.path.join(OUTDIR, 'supplement_pathway_coverage_scatter.png'),
            dpi=300, bbox_inches='tight')

coverage.sort_values('delta', ascending=False).to_csv(
    os.path.join(OUTDIR, 'supplement_pathway_coverage_scatter_sourcedata.csv'),
    index=False)

print(f'\nSaved the coverage panel and its supplement to {OUTDIR}')
