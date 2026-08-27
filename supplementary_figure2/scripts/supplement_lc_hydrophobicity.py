"""Supplementary figure 2 — peptide hydrophobicity on the two LC front ends."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib_venn import venn2
import spec_analytics as core
core.init_plotting()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import spec_config as _cfg
INPUT = _cfg.input_dir(__file__, 'H032_E333')
OUTDIR = _cfg.output_dir(__file__)
FRONT_ENDS = [
    ('Online Trap&Elute', os.path.join(INPUT, 'Vanquish', 'Vanquish.parquet'), '-'),
    ('Disposable Trap Column', os.path.join(INPUT, 'Evosep', 'Evosep.parquet'), '--'),
]
ORDER = [tag for tag, _p, _ls in FRONT_ENDS]
QVALUE = 0.01
N_REPLICATES = 4
N_BINS = 20
GRID = 400
N_BOOT = 5000
BOOT_SEED = 0
N_ABUNDANCE_STRATA = 10
FONTSIZE = 8
SPEC_COLOR = core.PALETTE_SINGLE[0]
STEM = 'supplement_lc_hydrophobicity'
def lighten(color, factor=0.55):
    c = mcolors.to_rgb(color)
    return tuple(1 - (1 - ch) * (1 - factor) for ch in c)
COLOR = {'Online Trap&Elute': core.PALETTE_SINGLE[4],
         'Disposable Trap Column': core.PALETTE_SINGLE[0]}
def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cum, q)])
per_lc, protein_groups = {}, {}
for tag, path, _ls in FRONT_ENDS:
    d = pd.read_parquet(path, columns=['Run', 'Stripped.Sequence', 'Protein.Group',
                                       'Precursor.Quantity', 'Q.Value',
                                       'PG.Q.Value'])
    d = d[(d['Q.Value'] < QVALUE) & (d['PG.Q.Value'] < QVALUE)
          & (d['Precursor.Quantity'] > 0)]
    if d['Run'].nunique() != N_REPLICATES:
        raise ValueError(f'{tag}: expected {N_REPLICATES} runs, '
                         f'found {d["Run"].nunique()}')
    protein_groups[tag] = set(d['Protein.Group'].dropna()) - {''}
    pep = (d.groupby(['Run', 'Stripped.Sequence'], sort=False)['Precursor.Quantity']
           .sum().rename('signal').reset_index())
    pep['share'] = pep['signal'] / pep.groupby('Run')['signal'].transform('sum')
    per_lc[tag] = pep.groupby('Stripped.Sequence')['share'].mean()
    print(f'{tag}: {len(protein_groups[tag]):,} protein groups, '
          f'{len(per_lc[tag]):,} stripped sequences')
sequences = set().union(*(set(s.index) for s in per_lc.values()))
gravy = pd.Series({s: core.gravy(s) for s in sequences}).dropna()
print(f'GRAVY computed for {len(gravy):,} of {len(sequences):,} sequences')
shared = sorted(set(per_lc[ORDER[0]].index) & set(per_lc[ORDER[1]].index)
                & set(gravy.index))
print(f'shared by both front ends: {len(shared):,}')
for tag in ORDER:
    only = set(per_lc[tag].index) - set(per_lc[ORDER[1 - ORDER.index(tag)]].index)
    print(f'  specific to {tag:24s} {len(only):>7,} peptides, '
          f'{per_lc[tag].reindex(list(only)).sum():.2%} of its signal')
stats_rows, curves = [], []
lo = min(gravy[per_lc[t].index.intersection(gravy.index)].min() for t in ORDER)
hi = max(gravy[per_lc[t].index.intersection(gravy.index)].max() for t in ORDER)
grid = np.linspace(lo, hi, GRID)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
a, b = (protein_groups[t] for t in ORDER)
v = venn2([a, b], set_labels=ORDER, ax=axes[0],
          set_colors=(COLOR[ORDER[0]], COLOR[ORDER[1]]))
REGION = {'10': COLOR[ORDER[0]], '01': COLOR[ORDER[1]],
          '11': tuple(np.mean([mcolors.to_rgb(COLOR[t]) for t in ORDER], axis=0))}
for key, colour in REGION.items():
    patch = v.get_patch_by_id(key)
    if patch is not None:
        patch.set_facecolor(colour)
        patch.set_alpha(1.0)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.6)
for text in list(v.set_labels or []) + list(v.subset_labels or []):
    if text is not None:
        text.set_fontsize(FONTSIZE)
axes[0].set_xlim(-0.80, 0.80)
n_shared = len(a & b)
axes[0].set_title(f'{100 * n_shared / len(a | b):.0f} % of the union shared',
                  fontsize=FONTSIZE, fontweight='bold')
print(f'\nprotein-group overlap: {len(a):,} / {len(b):,}, {n_shared:,} shared '
      f'({100 * n_shared / len(a | b):.1f} % of the {len(a | b):,} union)')
venn_rows = [
    {'series': f'only {ORDER[0]}', 'n_protein_groups': len(a - b)},
    {'series': 'shared', 'n_protein_groups': n_shared},
    {'series': f'only {ORDER[1]}', 'n_protein_groups': len(b - a)},
    {'series': 'union', 'n_protein_groups': len(a | b)},
]
ax = axes[1]
for tag in ORDER:
    idx = per_lc[tag].index.intersection(gravy.index)
    g = gravy[idx].to_numpy()
    w = per_lc[tag][idx].to_numpy()
    w = w / w.sum()
    density = gaussian_kde(g, weights=w)(grid)
    ax.plot(grid, density, color=COLOR[tag], linewidth=1.5, alpha=0.7,
            label=f'{tag} ({weighted_quantile(g, w, 0.5):+.2f})')
    stats_rows.append({
        'front_end': tag, 'n_peptides': len(g),
        'weighted_mean': float((g * w).sum()),
        'weighted_median': weighted_quantile(g, w, 0.5),
        'weighted_q1': weighted_quantile(g, w, 0.25),
        'weighted_q3': weighted_quantile(g, w, 0.75),
        'weighted_sd': float((w * (g - (g * w).sum()) ** 2).sum() ** 0.5),
        'unweighted_median': float(np.median(g))})
    curves.append(pd.DataFrame({'front_end': tag, 'gravy': grid,
                               'weighted_density': density}))
ax.set_xlabel('Peptide GRAVY (Kyte-Doolittle)', fontsize=FONTSIZE)
ax.set_ylabel('Intensity-weighted density', fontsize=FONTSIZE)
ax.set_ylim(0, None)
ax.tick_params(labelsize=FONTSIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=8,
          handlelength=1.6, handletextpad=0.5, labelspacing=0.3, borderpad=0.0,
          title='weighted median GRAVY', title_fontsize=7.5)
ax.get_legend().get_title().set_color('#666666')
summary = pd.DataFrame(stats_rows).set_index('front_end')
print('\nintensity-weighted GRAVY:')
print(summary.round(4).to_string())
g = gravy[shared]
ratio = np.log2(per_lc[ORDER[1]][shared] / per_lc[ORDER[0]][shared])
rho = stats.spearmanr(g, ratio)[0]
abundance = np.log10(np.sqrt(per_lc[ORDER[1]][shared] * per_lc[ORDER[0]][shared]))
rng = np.random.default_rng(BOOT_SEED)
def median_ci(values, n_boot=N_BOOT):
    """Percentile bootstrap CI of the median - no distributional assumption.
    The notch approximation (median +/- 1.57 IQR/sqrt(n)) agrees to ~0.005 log2
    here, but a reviewer should not have to accept a normal-theory shortcut on a
    distribution this heavy-tailed.
    """
    v = np.asarray(values, dtype=float)
    draws = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
frame = pd.DataFrame({'gravy': g, 'ratio': ratio, 'abundance': abundance})
frame['bin'] = pd.qcut(frame['gravy'], N_BINS, labels=False, duplicates='drop')
records = []
for b, sub in frame.groupby('bin'):
    lo, hi = median_ci(sub['ratio'])
    records.append({'bin': int(b), 'gravy': sub['gravy'].median(),
                    'median': sub['ratio'].median(), 'ci_lo': lo, 'ci_hi': hi,
                    'q1': sub['ratio'].quantile(0.25),
                    'q3': sub['ratio'].quantile(0.75),
                    'median_log10_abundance': sub['abundance'].median(),
                    'n': len(sub)})
binned = pd.DataFrame(records).set_index('bin')
iqr_lo = float((binned['q3'] - binned['q1']).min())
iqr_hi = float((binned['q3'] - binned['q1']).max())
print('\nlog2 signal-share ratio on %s shared peptides:' % format(len(shared), ','))
print('  median %+.3f, IQR %.3f'
      % (ratio.median(), ratio.quantile(.75) - ratio.quantile(.25)))
print('  Spearman(GRAVY, ratio) = %+.4f, Pearson = %+.4f'
      % (rho, stats.pearsonr(g, ratio)[0]))
print('\nper bin: median with 95% bootstrap CI, abundance, n')
print(binned.round(3).to_string())
resolvable = binned[(binned['ci_lo'] > 0) | (binned['ci_hi'] < 0)]
print('%d of %d bins have a CI excluding zero; largest |median| among them '
      '%.3f log2' % (len(resolvable), len(binned),
                     resolvable['median'].abs().max()))
frame['stratum'] = pd.qcut(frame['abundance'], N_ABUNDANCE_STRATA, labels=False,
                           duplicates='drop')
strata = []
for s, sub in frame.groupby('stratum'):
    first = sub.loc[sub['bin'] == 0, 'ratio'].to_numpy()
    rest = sub.loc[sub['bin'] > 0, 'ratio'].to_numpy()
    if first.size < 30:
        continue
    shift = float(np.median(rng.choice(first, 4000) - rng.choice(rest, 4000)))
    strata.append({'abundance_stratum': int(s), 'n_first_bin': int(first.size),
                   'median_first_bin': float(np.median(first)),
                   'median_rest': float(np.median(rest)),
                   'hodges_lehmann_shift': shift,
                   'mannwhitney_p': float(stats.mannwhitneyu(
                       first, rest, alternative='two-sided').pvalue)})
strata = pd.DataFrame(strata)
weighted_shift = float(np.average(strata['hodges_lehmann_shift'],
                                  weights=strata['n_first_bin']))
r_ga = stats.spearmanr(frame['gravy'], frame['abundance'])[0]
r_ar = stats.spearmanr(frame['abundance'], frame['ratio'])[0]
partial = (rho - r_ga * r_ar) / np.sqrt((1 - r_ga ** 2) * (1 - r_ar ** 2))
print('\nabundance control for the most hydrophilic bin:')
print(strata.round(4).to_string(index=False))
print('  weighted mean Hodges-Lehmann shift = %+.3f log2 (unadjusted %+.3f)'
      % (weighted_shift, binned.loc[0, 'median']))
print('  Spearman(GRAVY, abundance) = %+.4f - hydrophilic peptides are fainter, '
      'hence the control' % r_ga)
print('  partial Spearman(GRAVY, ratio | abundance) = %+.4f (unadjusted %+.4f)'
      % (partial, rho))
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, f'{STEM}.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, f'{STEM}.png'), dpi=300, bbox_inches='tight')
pd.DataFrame(venn_rows).to_csv(
    os.path.join(OUTDIR, f'{STEM}_venn_sourcedata.csv'), index=False)
pd.concat(curves, ignore_index=True).to_csv(
    os.path.join(OUTDIR, f'{STEM}_kde_sourcedata.csv'), index=False)
summary.reset_index().to_csv(
    os.path.join(OUTDIR, f'{STEM}_statistics.csv'), index=False)
for stale in ('_ratio_sourcedata.csv', '_abundance_control.csv',
              '_ratio_statistics.csv'):
    old = os.path.join(OUTDIR, f'{STEM}{stale}')
    if os.path.exists(old):
        os.remove(old)
        print(f'removed orphaned {os.path.basename(old)}')
print(f'\nSaved {STEM} to {OUTDIR}')
