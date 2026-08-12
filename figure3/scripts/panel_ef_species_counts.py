"""Figure 3e + 3f — identifications by species, single elution vs on-SPEC"""

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
INPUT = _cfg.input_dir(__file__)
REPORTS = {'Single elution': os.path.join(INPUT, 'LTH43', 'single_elution',
                                          'report.parquet'),
           'On-SPEC fractionation': os.path.join(INPUT, 'LTH43', 'fractionated',
                                                 'report.parquet')}
OUTDIR = _cfg.output_dir(__file__)
os.makedirs(OUTDIR, exist_ok=True)

QVALUE = 0.01
FRACTION_ROWS = ['A', 'B', 'C', 'D', 'E']
SPECIES = ['HUMAN', 'ARATH', 'YEAST']
SPECIES_LABEL = {'HUMAN': 'Human', 'ARATH': 'Arabidopsis', 'YEAST': 'Yeast'}
# Species hues avoid the method hues used in panel a (coral/sky/pink).
SPECIES_COLOR = {'HUMAN': core.PALETTE_SINGLE[1], 'ARATH': core.PALETTE_SINGLE[4],
                 'YEAST': core.PALETTE_SINGLE[3]}
BAR_WIDTH_IN = 0.38   # drawn bar width, matched across figure 3
POINT_SIZE = core.replicate_point_size(BAR_WIDTH_IN)
CONDITIONS = ['Single elution', 'On-SPEC fractionation']
# Rotated single-line labels: horizontal ones collide at 2.2 in panel width.
XTICK = {'Single elution': 'Single elution (1 run)',
         'On-SPEC fractionation': 'On-SPEC fractionation (5 runs)'}

# ---------------------------------------------------------------------------
# Per-replicate, per-species counts.
# ---------------------------------------------------------------------------
frames = []
for condition, path in REPORTS.items():
    x = pd.read_parquet(path, columns=[
        'Run', 'Precursor.Id', 'Protein.Group', 'Protein.Names', 'Decoy',
        'Q.Value', 'PG.Q.Value'])
    x = x[(x['Decoy'] == 0) & (x['Q.Value'] < QVALUE)
          & (x['PG.Q.Value'] < QVALUE)]
    frames.append(x.assign(condition=condition))
work = pd.concat(frames, ignore_index=True)

work['tag'] = work['Run'].str[-3:]
work['col'] = work['tag'].str[1:].astype(int)
# Column 01-03 for single elution, 04-06 for the fraction series; either way the
# column is the replicate. Fraction row F is absent from the fractionated search.
work['replicate'] = work['col'].where(work['col'] <= 3, work['col'] - 3)
work['species'] = work['Protein.Names'].str.extract(r'_(HUMAN|ARATH|YEAST)')[0]
if work['species'].isna().any():
    raise ValueError('rows without a species tag in Protein.Names')

runs_per_rep = work.groupby(['condition', 'replicate'])['tag'].nunique()
print('runs per replicate:'); print(runs_per_rep.to_string())

per_rep = (work.groupby(['condition', 'replicate', 'species'])
           .agg(protein_groups=('Protein.Group', 'nunique'),
                precursors=('Precursor.Id', 'nunique'))
           .reset_index())
totals = (work.groupby(['condition', 'replicate'])
          .agg(protein_groups=('Protein.Group', 'nunique'),
               precursors=('Precursor.Id', 'nunique')).reset_index())
print('\nper-replicate totals:'); print(totals.to_string(index=False))

means = per_rep.pivot_table(index='condition', columns='species',
                            values=['protein_groups', 'precursors'],
                            aggfunc='mean').reindex(CONDITIONS)
print('\nmean per replicate by species:'); print(means.round(0).to_string())
for metric in ('protein_groups', 'precursors'):
    t = totals.groupby('condition')[metric].mean().reindex(CONDITIONS)
    print(f'{metric}: {t.iloc[0]:,.0f} -> {t.iloc[1]:,.0f} '
          f'({100 * (t.iloc[1] / t.iloc[0] - 1):+.0f}%)')


def set_bar_width_inches(fig, ax, target_in=BAR_WIDTH_IN):
    """Rescale bars so their drawn width is `target_in` inches.

    Bar width in data units maps to a different physical width in every panel,
    because the axes width depends on how wide the tick labels are. Setting it
    from the rendered geometry keeps the bars identical across panels.
    """
    fig.canvas.draw()
    ax_w = ax.get_position().width * fig.get_size_inches()[0]
    x0, x1 = ax.get_xlim()
    w_data = target_in * (x1 - x0) / ax_w
    for patch in ax.patches:
        centre = patch.get_x() + patch.get_width() / 2
        patch.set_width(w_data)
        patch.set_x(centre - w_data / 2)


# ---------------------------------------------------------------------------
# Plot — one stacked bar per condition, species stacked, replicate points at
# each cumulative boundary.
# ---------------------------------------------------------------------------
def draw_panel(metric, ylabel, filename, show_legend):
    x = np.arange(len(CONDITIONS))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(2.2, 4))

    bottom = np.zeros(len(CONDITIONS))
    for sp in SPECIES:
        heights = np.array([
            per_rep.loc[(per_rep['condition'] == c) & (per_rep['species'] == sp),
                        metric].mean() for c in CONDITIONS])
        ax.bar(x, heights, 0.6, bottom=bottom, color=SPECIES_COLOR[sp],
               edgecolor='black', linewidth=0.5, label=SPECIES_LABEL[sp], zorder=2)
        bottom = bottom + heights

        # Per-replicate cumulative value at this species boundary.
        for i, c in enumerate(CONDITIONS):
            sub = per_rep[per_rep['condition'] == c]
            order = SPECIES[:SPECIES.index(sp) + 1]
            vals = (sub[sub['species'].isin(order)]
                    .groupby('replicate')[metric].sum().to_numpy())
            jit = rng.uniform(-0.07, 0.07, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jit, vals, s=POINT_SIZE, color='black',
                       alpha=0.75, linewidth=0.3, edgecolor='white', zorder=5)

    for i, c in enumerate(CONDITIONS):
        total = totals.loc[totals['condition'] == c, metric].mean()
        ax.text(i, bottom[i] * 1.015, f'{total:,.0f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([XTICK[c] for c in CONDITIONS], fontsize=8,
                       rotation=45, ha='right')
    ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    ax.set_ylim(0, bottom.max() * 1.14)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if show_legend:
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), frameon=False,
                  fontsize=7, handlelength=0.8, handleheight=0.9,
                  handletextpad=0.35, borderpad=0.0, labelspacing=0.25,
                  columnspacing=0.8, ncol=3)

    fig.tight_layout()
    set_bar_width_inches(fig, ax)
    fig.savefig(os.path.join(OUTDIR, f'{filename}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{filename}.png'), dpi=300, bbox_inches='tight')

    src = per_rep.assign(series='replicate value')[
        ['series', 'condition', 'replicate', 'species', metric]]
    bars = []
    for c in CONDITIONS:
        for sp in SPECIES:
            bars.append({'series': 'bar segment (mean)', 'condition': c,
                         'replicate': np.nan, 'species': sp,
                         metric: per_rep.loc[(per_rep['condition'] == c)
                                             & (per_rep['species'] == sp), metric].mean()})
        bars.append({'series': 'bar total (mean)', 'condition': c,
                     'replicate': np.nan, 'species': 'all',
                     metric: totals.loc[totals['condition'] == c, metric].mean()})
    pd.concat([src, pd.DataFrame(bars)], ignore_index=True).to_csv(
        os.path.join(OUTDIR, f'{filename}_sourcedata.csv'), index=False)


draw_panel('protein_groups', 'Protein groups', 'panel_e_protein_groups',
           show_legend=True)
draw_panel('precursors', 'Precursors', 'panel_f_precursors', show_legend=False)

print(f'\nSaved panels e and f to {OUTDIR}')
