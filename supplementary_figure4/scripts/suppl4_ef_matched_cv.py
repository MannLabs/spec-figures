"""Supplementary figure 4e, f — per-peptide precision at matched abundance."""
import os
import sys
import pandas as pd
import pyarrow.parquet as pq
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
CACHE = cs.cache_path('matched_cv_peptides.parquet')
SERIES = {
    'volume': dict(root=os.path.join(cs.FIG2_INPUT, 'H032_E306'),
                   levels={'5uL': 5, '10uL': 10, '40uL': 40, '100uL': 100,
                           '200uL': 200},
                   stem=lambda lv: f'{lv}.parquet', xlabel='Sample volume [µL]',
                   exclude=3),
    'input': dict(root=os.path.join(cs.FIG2_INPUT, 'H032_E305'),
                  levels={'5ng': 5, '20ng': 20, '50ng': 50, '200ng': 200,
                          '500ng': 500},
                  stem=lambda lv: 'report.parquet',
                  xlabel='Protein input amount [ng]', exclude=None),
}
METHODS = {'SPEC': 'SAX SPEC', 'PAC': 'PAC', 'ISD+': 'ISD+'}
COLOR = {'SAX SPEC': core.PALETTE_SINGLE[0], 'PAC': core.PALETTE_SINGLE[2],
         'ISD+': core.PALETTE_SINGLE[5]}
COLS = ['Run', 'Stripped.Sequence', 'Precursor.Quantity', 'Q.Value',
        'PG.Q.Value', 'Decoy']
N_DECILES = 10
PANELS = [
    ('suppl4_e_matched_cv_volume', 'volume', 5, '200 ng in 5 µL'),
    ('suppl4_f_matched_cv_input', 'input', 5, '5 ng input'),
]
PANEL_SIZE = (cf.PANEL_W_IN, 4)
def peptide_cv(path, exclude=None):
    """CV and mean intensity per peptide, within one condition's replicates."""
    df = pq.read_table(path, columns=COLS).to_pandas()
    if exclude is not None:
        runs = sorted(df['Run'].unique())
        if len(runs) >= exclude:
            df = df[df['Run'] != runs[exclude - 1]]
    df = df[(df['Q.Value'] < cs.QVALUE) & (df['PG.Q.Value'] < cs.QVALUE) &
            (df['Decoy'] == 0) & (df['Precursor.Quantity'] > 0)]
    pep = (df.groupby(['Run', 'Stripped.Sequence'], observed=True)
             ['Precursor.Quantity'].sum().reset_index())
    w = pep.pivot(index='Stripped.Sequence', columns='Run',
                  values='Precursor.Quantity')
    n = w.notna().sum(1)
    out = pd.DataFrame({'seq': w.index, 'n_obs': n.values,
                        'mean_int': w.mean(axis=1).values,
                        'cv': (100 * w.std(axis=1, ddof=1) / w.mean(axis=1)).values})
    return out[out.n_obs >= cs.MIN_VALUES_FOR_CV].reset_index(drop=True)
def build():
    rows = []
    for sname, spec in SERIES.items():
        for folder, level in spec['levels'].items():
            for folder_name, label in METHODS.items():
                path = os.path.join(spec['root'], folder_name, folder,
                                    spec['stem'](folder))
                t = peptide_cv(path, spec['exclude'])
                t['series'], t['level'], t['method'] = sname, level, label
                rows.append(t)
            print(f'  {sname} {folder} done')
    d = pd.concat(rows, ignore_index=True)
    os.makedirs(cs.CACHE_DIR, exist_ok=True)
    d.to_parquet(CACHE, index=False)
    return d
d = pd.read_parquet(CACHE) if os.path.exists(CACHE) else build()
shared, deciles = [], []
for (sname, level), sub in d.groupby(['series', 'level'], sort=False):
    sets = {m: set(g.seq) for m, g in sub.groupby('method')}
    common = set.intersection(*sets.values())
    s = sub[sub.seq.isin(common)]
    rank = s.groupby('seq').mean_int.mean().rank(pct=True) * N_DECILES
    s = s.assign(dec=s.seq.map(rank.clip(upper=N_DECILES - 1e-9)).astype(int))
    for m, g in s.groupby('method'):
        shared.append(dict(series=sname, level=level, method=m,
                           n_shared=len(common), median_cv=g.cv.median(),
                           median_cv_all=sub[sub.method == m].cv.median(),
                           n_all=int((sub.method == m).sum())))
        for dec, gg in g.groupby('dec'):
            deciles.append(dict(series=sname, level=level, method=m,
                                decile=int(dec) + 1, median_cv=gg.cv.median(),
                                n=len(gg)))
sh, dec = pd.DataFrame(shared), pd.DataFrame(deciles)
print('median CV [%] on the shared peptide set (pooled value in brackets):')
for sname, spec in SERIES.items():
    print(f'  --- {sname}')
    for lv in spec['levels'].values():
        parts = []
        for m in METHODS.values():
            r = sh[(sh.series == sname) & (sh.level == lv) & (sh.method == m)]
            if len(r):
                r = r.iloc[0]
                parts.append(f'{m} {r.median_cv:5.1f} ({r.median_cv_all:5.1f})')
        n = sh[(sh.series == sname) & (sh.level == lv)].n_shared.iloc[0]
        print(f'    {lv:>5}  ' + '   '.join(parts) + f'   n shared {n:,}')
for stem, sname, level, title in PANELS:
    s = dec[(dec.series == sname) & (dec.level == level)]
    n_shared = int(sh[(sh.series == sname) & (sh.level == level)]
                   .n_shared.iloc[0])
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    handles = {}
    for m in ('PAC', 'ISD+', 'SAX SPEC'):
        g = s[s.method == m].sort_values('decile')
        handles[m], = ax.plot(g.decile, g.median_cv, color=COLOR[m], lw=1.6,
                              marker='o', ms=4, mec='white', mew=0.7, label=m)
    ax.set_title(f'{title}\nn = {n_shared:,} shared peptides',
                 fontsize=cs.FONTSIZE, fontweight='bold')
    ax.set_xlabel('Abundance decile', fontsize=cs.FONTSIZE)
    ax.set_ylabel('Median peptide CV [%]', fontsize=cs.FONTSIZE)
    ax.set_xticks(range(1, N_DECILES + 1, 3))
    ax.tick_params(labelsize=cs.FONTSIZE)
    ax.set_ylim(0, s.median_cv.max() * 1.42)
    ax.legend([handles[m] for m in ('SAX SPEC', 'PAC', 'ISD+')],
              ['SAX SPEC', 'PAC', 'ISD+'], frameon=False,
              fontsize=cs.LEGEND_FONTSIZE, loc='upper right', handlelength=1.2,
              labelspacing=0.3, borderpad=0.2, ncol=3, columnspacing=0.9,
              handletextpad=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f'{stem}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{stem}.png'), dpi=300,
                bbox_inches='tight')
    rows = [{'series': 'decile point (drawn)', 'experiment': r.series,
             'level': r.level, 'method': r.method, 'decile': r.decile,
             'median_cv_pct': round(r.median_cv, 2),
             'median_cv_pooled_pct': '', 'n_peptides': r.n}
            for r in s.itertuples(index=False)]
    rows += [{'series': 'shared-set summary (all levels)',
              'experiment': r.series, 'level': r.level, 'method': r.method,
              'decile': '', 'median_cv_pct': round(r.median_cv, 2),
              'median_cv_pooled_pct': round(r.median_cv_all, 2),
              'n_peptides': r.n_shared}
             for r in sh.itertuples(index=False)]
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTDIR, f'{stem}_sourcedata.csv'), index=False)
    print(f'Saved {stem} to {OUTDIR}')
