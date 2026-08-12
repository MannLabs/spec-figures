"""Mixed-species benchmarks — species assignment, composition and ratio recovery.

A mixed-species experiment spikes two or more proteomes together at known
ratios and asks whether the platform recovers them. That needs three things
this module provides: labelling every protein with the organism it came from,
showing the design and what was identified from it, and plotting measured
against expected ratios.

Species come from the UniProt entry name — ``ZFP91_HUMAN``, ``EF1A_YEAST`` —
which the loaders keep in ``protein_names``. That suffix is the reliable
handle: accessions carry no organism, and the ``OS=`` description field is
free text that varies by FASTA release
(``Saccharomyces cerevisiae (strain ATCC 204508 / S288c)``).

**Protein groups that span organisms are dropped, not assigned.** Peptides
shared between proteomes group proteins across species, and such a group has
no defined expected ratio — it is a mixture. Silently taking the first
accession would place it at the wrong ratio and widen the very distribution
the benchmark is measuring. ``assign_species`` labels these ``'Ambiguous'``
and the plots exclude them by default, reporting how many.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .plotting._style import PALETTE_SINGLE, _resolve_panel_size

__all__ = [
    'SPECIES_SUFFIXES',
    'assign_species',
    'sum_precursors_to_protein',
    'species_ratio_table',
    'species_ratio_accuracy',
    'plot_expected_composition',
    'plot_species_counts',
    'plot_species_ratio',
    'plot_ratio_accuracy',
    'plot_species_cv_ecdf',
]

#: UniProt entry-name suffix -> display name, for the organisms that show up in
#: two- and three-proteome benchmarks. Pass your own mapping to `assign_species`
#: for anything else.
SPECIES_SUFFIXES = {
    'HUMAN': 'Human',
    'YEAST': 'Yeast',
    'ECOLI': 'E. coli',
    'MOUSE': 'Mouse',
    'ARATH': 'A. thaliana',
    'CAEEL': 'C. elegans',
    'DROME': 'D. melanogaster',
}

AMBIGUOUS = 'Ambiguous'
UNKNOWN = 'Unknown'


def assign_species(df, *, col='protein_names', suffixes=None, sep=';',
                   out_col='species'):
    """Label each row with the organism of its protein group.

    :param df: long DataFrame carrying `col`.
    :param str col: column holding UniProt entry names (``'ZFP91_HUMAN'``).
        Several names separated by `sep` are read as one protein group.
    :param dict suffixes: entry-name suffix -> display name; defaults to
        :data:`SPECIES_SUFFIXES`.
    :returns: a copy of `df` with `out_col` added.

    A group whose members resolve to more than one organism is labelled
    ``'Ambiguous'`` rather than assigned to one of them — see the module
    docstring. A name with no recognised suffix is ``'Unknown'``.
    """
    suffixes = SPECIES_SUFFIXES if suffixes is None else suffixes
    if col not in df.columns:
        raise KeyError(
            f'{col!r} not in df; species come from the UniProt entry name '
            f'(available: {sorted(df.columns)[:12]}...)')

    # Resolve once per distinct protein-group string, not once per row — the
    # long frame repeats each group across runs and precursors.
    unique = pd.Series(df[col].dropna().unique())
    lookup = {}
    for value in unique:
        found = set()
        for name in str(value).split(sep):
            name = name.strip()
            suffix = name.rsplit('_', 1)[-1].upper() if '_' in name else ''
            found.add(suffixes.get(suffix, UNKNOWN))
        found.discard(UNKNOWN)
        if len(found) == 1:
            lookup[value] = found.pop()
        elif len(found) > 1:
            lookup[value] = AMBIGUOUS
        else:
            lookup[value] = UNKNOWN

    out = df.copy()
    out[out_col] = out[col].map(lookup).fillna(UNKNOWN)
    return out


def _condition_runs(sample_info, group_col):
    return {c: set(g['run']) for c, g in sample_info.groupby(group_col)}


def _pg_matrix(df, runs, value_col='pg_intensity', id_col='protein_group'):
    """One row per protein group, one column per run, intensities > 0."""
    sub = df[df['run'].isin(runs)].dropna(subset=[id_col, value_col])
    sub = sub[sub[value_col] > 0]
    # A protein group repeats once per precursor; collapse to one value per run.
    return sub.pivot_table(index=id_col, columns='run', values=value_col,
                           aggfunc='first')


def sum_precursors_to_protein(df, runs, *, id_col='protein_group',
                              value_col='precursor_intensity',
                              precursor_col='precursor_id',
                              min_completeness=1.0):
    """Protein quant by summing precursor intensities, on complete precursors.

    An alternative to MaxLFQ that is transparent about what it did: the
    protein's value is the sum of its precursors, full stop.

    **Completeness is enforced on the precursors, before the sum.** This is the
    whole point of doing it here rather than with a groupby. If a precursor is
    missing in one run, summing whatever is present gives that run a total over
    a smaller set of precursors, so the protein looks lower there for a reason
    that has nothing to do with abundance. Across a dilution series that biases
    every ratio toward the more completely observed condition. Requiring a
    precursor in every compared run first means the sums are over identical
    sets and the ratio is a ratio of like for like.

    The trade is coverage: at ``min_completeness=1.0`` a protein keeps only its
    precursors seen everywhere, and a protein with none is dropped entirely.
    That is the right trade for a ratio measurement and the wrong one for an
    identification count — use :func:`plot_species_counts` for the latter.

    :param runs: the runs to include, all of them treated as one block.
    :returns: DataFrame indexed by protein group, one column per run.
    """
    sub = df[df['run'].isin(runs)].dropna(subset=[id_col, value_col,
                                                  precursor_col])
    sub = sub[sub[value_col] > 0]
    wide = sub.pivot_table(index=precursor_col, columns='run',
                           values=value_col, aggfunc='max')
    wide = wide.reindex(columns=list(runs))
    keep = wide.notna().sum(axis=1) >= min_completeness * len(runs)
    wide = wide[keep]

    pg = (sub.drop_duplicates(precursor_col)
          .set_index(precursor_col)[id_col].reindex(wide.index))
    out = wide.groupby(pg).sum(min_count=1)
    return out.where(out > 0)


def species_ratio_table(df, sample_info, condition_a, condition_b, *,
                        group_col='condition2', value_col='pg_intensity',
                        id_col='protein_group', species_col='species',
                        min_completeness=1.0, max_cv=None,
                        drop_ambiguous=True, quant='column',
                        precursor_col='precursor_id',
                        normalize_species=None):
    """Per-protein measured log2 ratio between two conditions.

    Ratios and the intensity axis are both taken as **means of log2** — the
    geometric mean — so the plotted effect size lives in the same space as the
    axis it is drawn on. See the package's log-space-means convention.

    :param float min_completeness: fraction of each condition's runs in which
        the protein must be quantified (1.0 = every run of both).
    :param float max_cv: optional upper bound on the within-condition CV,
        applied to both conditions.
    :param str quant: ``'column'`` (default) reads `value_col` as an
        already-computed protein value (MaxLFQ). ``'sum_precursors'`` builds
        the protein value here by summing `value_col` over complete precursors
        across all compared runs — see :func:`sum_precursors_to_protein`.
    :param str normalize_species: shift each run's log2 values by the median of
        this species, which the design holds constant, so its ratio is expected
        to be 0. Removes a loading difference between runs *before* the
        per-condition means are taken, which is stricter than correcting the
        expected ratios afterwards (:func:`plot_species_ratio`) because it also
        removes run-to-run spread within a condition. Prefer it when the
        conditions are being compared quantitatively; None to leave the data
        untouched.
    :returns: DataFrame with `id_col`, `species`, `log2_intensity` (mean over
        both conditions), `log2_ratio` (a - b), per-condition means, CVs and
        completeness.
    """
    runs = _condition_runs(sample_info, group_col)
    for cond in (condition_a, condition_b):
        if cond not in runs:
            raise ValueError(f'condition {cond!r} not in '
                             f'sample_info[{group_col!r}] '
                             f'(available: {sorted(runs)})')

    species = (df[[id_col, species_col]].drop_duplicates()
               .set_index(id_col)[species_col])

    if quant == 'sum_precursors':
        # Completeness spans BOTH conditions' runs: the precursors summed must
        # be the same set on either side of the ratio, not merely complete
        # within each condition separately.
        all_runs = list(runs[condition_a]) + list(runs[condition_b])
        combined = sum_precursors_to_protein(
            df, all_runs, id_col=id_col, value_col=value_col,
            precursor_col=precursor_col, min_completeness=min_completeness)
        matrices = {tag: combined[list(runs[cond])]
                    for tag, cond in (('a', condition_a), ('b', condition_b))}
    elif quant == 'column':
        matrices = {tag: _pg_matrix(df, runs[cond], value_col, id_col)
                    for tag, cond in (('a', condition_a), ('b', condition_b))}
    else:
        raise ValueError("quant must be 'column' or 'sum_precursors', "
                         f'got {quant!r}')

    if normalize_species is not None:
        wide = pd.concat(matrices.values(), axis=1)
        log2_all = np.log2(wide)
        is_ref = log2_all.index.map(species).to_series(
            index=log2_all.index).eq(normalize_species)
        if not is_ref.any():
            raise ValueError(
                f'normalize_species={normalize_species!r} matches no protein '
                'group; nothing to normalise against')
        shift = log2_all.loc[is_ref.to_numpy()].median(axis=0)
        matrices = {tag: 2.0 ** (np.log2(m).sub(shift[m.columns], axis=1))
                    for tag, m in matrices.items()}

    stats = {}
    for tag, cond in (('a', condition_a), ('b', condition_b)):
        mat = matrices[tag]
        n_runs = len(runs[cond])
        log2 = np.log2(mat)
        stats[tag] = pd.DataFrame({
            f'mean_log2_{tag}': log2.mean(axis=1),
            f'cv_{tag}': mat.std(axis=1, ddof=1) / mat.mean(axis=1),
            f'completeness_{tag}': mat.notna().sum(axis=1) / n_runs,
        })

    out = stats['a'].join(stats['b'], how='inner')
    out['log2_ratio'] = out['mean_log2_a'] - out['mean_log2_b']
    out['log2_intensity'] = (out['mean_log2_a'] + out['mean_log2_b']) / 2
    out = out.reset_index().rename(columns={'index': id_col})
    out[species_col] = out[id_col].map(species).fillna(UNKNOWN)

    n_before = len(out)
    keep = ((out['completeness_a'] >= min_completeness)
            & (out['completeness_b'] >= min_completeness))
    if max_cv is not None:
        keep &= (out['cv_a'].fillna(0) <= max_cv) & (out['cv_b'].fillna(0) <= max_cv)
    if drop_ambiguous:
        keep &= ~out[species_col].isin([AMBIGUOUS, UNKNOWN])
    out = out[keep].reset_index(drop=True)
    n_ambiguous = int((~keep).sum())
    out.attrs['n_before_filter'] = n_before
    out.attrs['n_dropped'] = n_ambiguous
    return out


def expected_log2_ratios(composition_a, composition_b, *, reference=None,
                         offset=0.0):
    """Nominal log2 ratio per species from two mixing designs.

    :param dict composition_a: species -> fraction (or percentage) in A.
    :param dict composition_b: the same for B.
    :param float offset: added to every expected ratio. Use it to absorb a
        loading difference between the two samples — see `plot_species_ratio`.
    """
    species = sorted(set(composition_a) | set(composition_b))
    out = {}
    for s in species:
        a, b = composition_a.get(s, 0.0), composition_b.get(s, 0.0)
        out[s] = np.log2(a / b) + offset if a > 0 and b > 0 else np.nan
    return out


def species_ratio_accuracy(df, sample_info, pairs, compositions, *,
                           species_order=None, verbose=True, **table_kwargs):
    """Measured against expected log2 ratio, over several condition pairs.

    One point per (pair, species). A single pair gives three points spanning
    whatever ratios that pair happens to encode; a set of pairs spanning the
    design covers the fold-change range properly, which is what makes the
    slope of measured-against-expected interpretable.

    That slope is the **ratio compression** of the pipeline: a value below 1
    means every fold change is understated by the same factor, which is the
    normal behaviour of a DIA quantification with co-isolated background.
    Because it is one number across the whole range, it can be divided out —
    see :func:`plot_ratio_accuracy` with ``correct=True``.

    :param pairs: iterable of ``(condition_a, condition_b)``.
    :param dict compositions: ``{condition: {species: fraction}}``, the design.
    :param table_kwargs: passed to :func:`species_ratio_table` (``quant``,
        ``normalize_species``, ``min_completeness``, ...).
    :returns: ``(accuracy, fit)``. `accuracy` has one row per (pair, species)
        with expected, observed median and mean, sd and n. `fit` is a dict with
        ``slope``, ``intercept``, ``r_squared`` and ``n_points``.

    The observed value is the **median** over proteins, not the mean: a
    mixed-species ratio distribution has an interference tail on the side of
    the constant background, and the mean follows it.
    """
    rows = []
    for cond_a, cond_b in pairs:
        table = species_ratio_table(df, sample_info, cond_a, cond_b,
                                    **table_kwargs)
        species = (species_order if species_order is not None
                   else sorted(table['species'].unique()))
        expected = expected_log2_ratios(compositions[cond_a],
                                        compositions[cond_b])
        for s in species:
            v = table.loc[table['species'] == s, 'log2_ratio'].dropna()
            if v.empty or not np.isfinite(expected.get(s, np.nan)):
                continue
            rows.append({'condition_a': cond_a, 'condition_b': cond_b,
                         'species': s, 'expected': float(expected[s]),
                         'observed': float(v.median()),
                         'observed_mean': float(v.mean()),
                         'sd': float(v.std(ddof=1)), 'n': int(len(v))})
        if verbose:
            print(f'  {cond_a} vs {cond_b}: ' + '  '.join(
                f'{r["species"]} {r["observed"]:+.3f}/{r["expected"]:+.3f}'
                for r in rows if r['condition_a'] == cond_a
                and r['condition_b'] == cond_b))

    accuracy = pd.DataFrame(rows)
    fit = _fit_line(accuracy['expected'].to_numpy(),
                    accuracy['observed'].to_numpy())
    if verbose:
        print(f'  compression slope {fit["slope"]:.3f}, '
              f'intercept {fit["intercept"]:+.3f}, R² {fit["r_squared"]:.4f}')
    return accuracy, fit


def _fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {'slope': float(slope), 'intercept': float(intercept),
            'r_squared': 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            'rms_residual': float(np.sqrt(ss_res / len(y))),
            'n_points': int(len(y))}


def plot_ratio_accuracy(accuracy, fit, *, correct=False, species_order=None,
                        palette=None, figsize=None, title=None,
                        label_fontsize=10, tick_fontsize=10,
                        legend_fontsize=8, ax=None):
    """Measured against expected log2 ratio, with the identity line.

    :param bool correct: divide every observed ratio by ``fit['slope']``,
        removing the global compression.

        **The corrected panel's slope is 1.000 by construction, so do not
        report it as a result.** Dividing the points by the slope fitted to
        them and refitting can only return 1. What the correction demonstrates
        is that a *single* factor suffices across the whole fold-change range;
        the evidence for that is R² and the residual scatter, both unchanged by
        the rescaling. Quote the compression factor itself as the finding, and
        the residual RMS as how well one factor describes it.

    :returns: ``(fig, ax, source_df)``.
    """
    import matplotlib.pyplot as plt

    acc = accuracy.copy()
    slope = fit['slope']
    if correct:
        acc['observed'] = acc['observed'] / slope
        shown = _fit_line(acc['expected'].to_numpy(), acc['observed'].to_numpy())
    else:
        shown = fit

    species = (list(species_order) if species_order is not None
               else sorted(acc['species'].unique()))
    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(species)}

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    lo = min(acc['expected'].min(), acc['observed'].min()) - 0.4
    hi = max(acc['expected'].max(), acc['observed'].max()) + 0.4
    ax.plot([lo, hi], [lo, hi], ls='--', color='#888888', lw=1, zorder=1,
            label='y = x')
    for s in species:
        sub = acc[acc['species'] == s]
        ax.scatter(sub['expected'], sub['observed'], s=55, color=palette[s],
                   edgecolor='black', linewidth=0.6, zorder=3, label=s)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Expected log₂ ratio', fontsize=label_fontsize)
    ax.set_ylabel(('Measured log₂ ratio, corrected' if correct
                   else 'Measured log₂ ratio'), fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.tick_params(labelsize=tick_fontsize)

    # After correction the slope is tautologically 1, so report what is not:
    # how tightly one factor describes every pair.
    note = (f'compression {slope:.3f}\n'
            f'residual RMS {shown["rms_residual"]:.3f}\n'
            f'R² {shown["r_squared"]:.4f}') if correct else (
        f'slope {shown["slope"]:.3f}\n'
        f'intercept {shown["intercept"]:+.3f}\n'
        f'R² {shown["r_squared"]:.4f}')
    ax.text(0.97, 0.03, note, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=legend_fontsize, family='monospace')
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='upper left')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    acc['fit_slope'] = shown['slope']
    acc['fit_intercept'] = shown['intercept']
    acc['compression_slope'] = slope
    return fig, ax, acc


def plot_species_cv_ecdf(df, sample_info, condition, *, group_col='condition2',
                         species_col='species', id_col='protein_group',
                         value_col='pg_intensity', min_values=3,
                         species_order=None, palette=None, figsize=None,
                         cv_threshold=20.0, xlim=(0, 40), title=None,
                         label_fontsize=10, tick_fontsize=10,
                         legend_fontsize=8, ax=None):
    """ECDF of per-protein CV within one condition, one curve per species.

    Splitting by species is the point: in a mixed-species design the three
    proteomes sit at different loads in the same run, so their CVs report
    precision at three abundance levels from a single sample. A pooled CV curve
    averages that away.

    :returns: ``(fig, ax, source_df)`` with the plotted ECDF steps.
    """
    import matplotlib.pyplot as plt

    runs = sample_info.loc[sample_info[group_col] == condition, 'run']
    if runs.empty:
        raise ValueError(f'condition {condition!r} not in '
                         f'sample_info[{group_col!r}]')
    mat = _pg_matrix(df, set(runs), value_col, id_col)
    mat = mat[mat.notna().sum(axis=1) >= min_values]
    # CV on linear intensities, matching the package convention.
    cv = (mat.std(axis=1, ddof=1) / mat.mean(axis=1) * 100).rename('cv_pct')

    species = (df[[id_col, species_col]].drop_duplicates()
               .set_index(id_col)[species_col])
    frame = pd.DataFrame({'cv_pct': cv,
                          'species': cv.index.map(species)}).dropna()
    frame = frame[~frame['species'].isin([AMBIGUOUS, UNKNOWN])]

    order = (list(species_order) if species_order is not None
             else sorted(frame['species'].unique()))
    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(order)}

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    rows = []
    for s in order:
        v = np.sort(frame.loc[frame['species'] == s, 'cv_pct'].to_numpy())
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1) / v.size
        median = float(np.median(v))
        below = float((v < cv_threshold).mean())
        ax.plot(v, y, color=palette[s], lw=1.8,
                label=f'{s} ({median:.1f}%)')
        rows.extend({'species': s, 'cv_pct': float(c),
                     'cumulative_fraction': float(f)} for c, f in zip(v, y))
        print(f'    {s:<8} median CV {median:.1f}%   '
              f'{below:.1%} below {cv_threshold:g}%   n = {v.size:,}')

    ax.axvline(cv_threshold, ls=':', color='#d62728', lw=1.2, alpha=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.0)
    # Every 0.1 so the "90% below X" reading can be taken off the axis.
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.set_xlabel('Protein CV [%]', fontsize=label_fontsize)
    ax.set_ylabel('Cumulative fraction', fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.tick_params(labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='lower right',
              title='median CV', title_fontsize=legend_fontsize)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_expected_composition(compositions, *, species_order=None,
                              condition_order=None, palette=None,
                              figsize=None, title=None,
                              y_label='Percentage of protein amounts',
                              x_label='Sample', label_fontsize=10,
                              tick_fontsize=10, legend_fontsize=8,
                              value_fontsize=8, ax=None):
    """Stacked bars of the designed species composition per condition.

    :param dict compositions: ``{condition: {species: percentage}}``.
    :returns: ``(fig, ax, source_df)``.
    """
    conditions = (list(compositions) if condition_order is None
                  else list(condition_order))
    species = species_order or sorted(
        {s for c in compositions.values() for s in c}, reverse=True)
    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(species)}
    if isinstance(palette, (list, tuple)):
        palette = {s: palette[i % len(palette)] for i, s in enumerate(species)}

    import matplotlib.pyplot as plt
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    rows = []
    bottom = np.zeros(len(conditions))
    x = np.arange(len(conditions))
    for s in species:
        vals = np.array([compositions[c].get(s, 0.0) for c in conditions],
                        dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.7, label=s,
               color=palette[s], edgecolor='none')
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(xi, b + v / 2, f'{v:g}', ha='center', va='center',
                        fontsize=value_fontsize)
            rows.append({'condition': conditions[xi], 'species': s,
                         'percentage': float(v)})
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right',
                       fontsize=tick_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylim(0, max(bottom) * 1.02)
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_species_counts(df, sample_info, *, group_col='condition2',
                        species_col='species', id_col='protein_group',
                        value_col='pg_intensity', condition_order=None,
                        species_order=None, palette=None, figsize=None,
                        title=None, y_label='Number of proteins',
                        label_fontsize=10, tick_fontsize=10,
                        legend_fontsize=8, value_fontsize=7,
                        count_basis='detected', evidence_col='precursor_intensity',
                        drop_ambiguous=True, ax=None):
    """Stacked bars of identifications per species per condition.

    A protein counts for a condition when it has evidence in at least one of
    that condition's runs. What counts as evidence is the `count_basis`:

    ``'detected'``   (default) any precursor of the protein was quantified,
                     i.e. `evidence_col` > 0. This is "identified", and it is
                     the more sensitive of the two — a low-abundance protein
                     can have observed peptides without surviving protein-level
                     quantification.
    ``'quantified'`` the protein itself carries a `value_col` (MaxLFQ) value.
                     Stricter, and the right basis when the panel sits next to
                     something that uses those intensities.

    The two diverge most where it matters in a dilution design: in a
    three-proteome mix the least abundant organism lost roughly half its count
    under ``'quantified'`` at the lowest spike-in, because MaxLFQ needs more
    evidence than detection does. Say which one a figure reports.
    """
    import matplotlib.pyplot as plt

    if count_basis not in ('detected', 'quantified'):
        raise ValueError("count_basis must be 'detected' or 'quantified', "
                         f'got {count_basis!r}')
    basis_col = evidence_col if count_basis == 'detected' else value_col
    work = df.dropna(subset=[basis_col])
    work = work[work[basis_col] > 0]
    if drop_ambiguous:
        work = work[~work[species_col].isin([AMBIGUOUS, UNKNOWN])]
    work = work.merge(sample_info[['run', group_col]], on='run', how='left')

    counts = (work.groupby([group_col, species_col])[id_col]
              .nunique().unstack(fill_value=0))
    conditions = list(condition_order) if condition_order is not None else list(counts.index)
    species = list(species_order) if species_order is not None else list(counts.columns)
    counts = counts.reindex(index=conditions, columns=species, fill_value=0)

    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(species)}
    if isinstance(palette, (list, tuple)):
        palette = {s: palette[i % len(palette)] for i, s in enumerate(species)}

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    x = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    rows = []
    for s in species:
        vals = counts[s].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.7, label=s, color=palette[s],
               edgecolor='none')
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(xi, b + v / 2, f'{int(v):,}', ha='center', va='center',
                        fontsize=value_fontsize, color='white')
            rows.append({'condition': conditions[xi], 'species': s,
                         'n': int(v)})
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right',
                       fontsize=tick_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    if title:
        ax.set_title(title, fontsize=label_fontsize + 1, fontweight='bold')
    ax.legend(fontsize=legend_fontsize, frameon=False, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if created:
        fig.tight_layout()
    return fig, ax, pd.DataFrame(rows)


def plot_species_ratio(df, sample_info, condition_a, condition_b, *,
                       composition_a=None, composition_b=None,
                       expected_ratios=None, reference_species='Human',
                       group_col='condition2', species_col='species',
                       id_col='protein_group', value_col='pg_intensity',
                       min_completeness=1.0, max_cv=None,
                       species_order=None, palette=None,
                       figsize=(12, 4), width_ratios=(6, 1, 1),
                       ylim=None, point_size=6, point_alpha=0.5,
                       label_fontsize=10, tick_fontsize=10,
                       legend_fontsize=8, rasterize_over=7000,
                       **table_kwargs):
    """Ratio-recovery panel: MA-style scatter, marginal density, and boxplots.

    Three axes side by side, sharing the log2-ratio axis:

    * measured log2 ratio against mean log2 intensity, coloured by species,
      with the expected ratio of each species as a dashed line;
    * the marginal density per species, annotated with its s.d.;
    * a boxplot per species.

    **The expected ratios absorb a loading offset.** Nominal ratios come from
    ``composition_a`` / ``composition_b``, but the two samples are rarely
    loaded at exactly the same total, so every measured ratio carries a common
    shift. It is estimated as the median measured ratio of
    ``reference_species`` — the proteome held constant by design, whose ratio
    is therefore expected to be 0 — and added to all expected values. This
    shifts the *reference lines* and leaves the measured data untouched, so the
    scatter still shows the ratios as measured. Pass ``reference_species=None``
    to compare against the nominal ratios directly, or ``expected_ratios=``
    to set the lines yourself.

    :returns: ``(fig, axes, stats, ratio_df)`` — `stats` has n, median, mean,
        s.d. and the expected ratio per species.
    """
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    ratio_df = species_ratio_table(
        df, sample_info, condition_a, condition_b, group_col=group_col,
        value_col=value_col, id_col=id_col, species_col=species_col,
        min_completeness=min_completeness, max_cv=max_cv, **table_kwargs)

    species = (list(species_order) if species_order is not None
               else list(ratio_df[species_col].value_counts().index))
    palette = palette or {s: PALETTE_SINGLE[i % len(PALETTE_SINGLE)]
                          for i, s in enumerate(species)}
    if isinstance(palette, (list, tuple)):
        palette = {s: palette[i % len(palette)] for i, s in enumerate(species)}

    # Expected ratios, plus the loading offset read off the reference species.
    offset = 0.0
    if expected_ratios is None:
        if composition_a is None or composition_b is None:
            raise ValueError('pass either expected_ratios= or both '
                             'composition_a= and composition_b=')
        if reference_species is not None:
            ref = ratio_df.loc[ratio_df[species_col] == reference_species,
                               'log2_ratio']
            if ref.empty:
                raise ValueError(
                    f'reference_species={reference_species!r} has no proteins '
                    'after filtering; pass reference_species=None to skip the '
                    'loading-offset correction')
            offset = float(ref.median())
        expected_ratios = expected_log2_ratios(composition_a, composition_b,
                                               offset=offset)

    fig, axes = plt.subplots(
        1, 3, figsize=figsize, sharey=True,
        gridspec_kw={'width_ratios': list(width_ratios), 'wspace': 0.06})
    ax_scatter, ax_density, ax_box = axes

    for s in species:
        sub = ratio_df[ratio_df[species_col] == s]
        coll = ax_scatter.scatter(
            sub['log2_intensity'], sub['log2_ratio'], s=point_size,
            color=palette[s], alpha=point_alpha, edgecolor='none', label=s)
        # Keep text and axes vector; only the cloud is rasterized, and only
        # once it is dense enough to slow an editor down.
        if len(sub) > rasterize_over:
            coll.set_rasterized(True)

    for s in species:
        exp = expected_ratios.get(s, np.nan)
        if np.isfinite(exp):
            ax_scatter.axhline(exp, color='#444444', ls='--', lw=0.8, zorder=0)
            n = int((ratio_df[species_col] == s).sum())
            ax_scatter.annotate(
                f'n = {n:,}', xy=(0.62, exp), xycoords=('axes fraction', 'data'),
                va='bottom', ha='left', fontsize=legend_fontsize)

    rows = []
    for s in species:
        v = ratio_df.loc[ratio_df[species_col] == s, 'log2_ratio'].dropna()
        if v.empty:
            continue
        # gaussian_kde raises on a degenerate sample (one protein, or every
        # ratio identical). Nothing to smooth there, so draw a rule at the value
        # and still report the species in `stats`.
        if len(v) < 3 or np.isclose(v.std(ddof=0), 0):
            ax_density.axhline(float(v.iloc[0]), color=palette[s], lw=1.0)
        else:
            grid = np.linspace(v.min(), v.max(), 512)
            dens = sps.gaussian_kde(v)(grid)
            ax_density.plot(dens, grid, color=palette[s], lw=1.0)
            ax_density.fill_betweenx(grid, 0, dens, color=palette[s], alpha=0.25)
        rows.append({'species': s, 'n': int(len(v)), 'median': float(v.median()),
                     'mean': float(v.mean()), 'sd': float(v.std(ddof=1)),
                     'expected_log2_ratio': float(expected_ratios.get(s, np.nan)),
                     'loading_offset_log2': offset})

    stats = pd.DataFrame(rows)
    for _, r in stats.iterrows():
        ax_density.annotate(
            f'{r["sd"]:.2f}', xy=(0.95, r['expected_log2_ratio']),
            xycoords=('axes fraction', 'data'), ha='right', va='bottom',
            fontsize=legend_fontsize, color=palette[r['species']])
    ax_density.set_title('s.d.', fontsize=label_fontsize, loc='right')

    box_data = [ratio_df.loc[ratio_df[species_col] == s, 'log2_ratio'].dropna()
                for s in species]
    bp = ax_box.boxplot(box_data, positions=range(len(species)), widths=0.6,
                        patch_artist=True, showcaps=True,
                        flierprops=dict(marker='o', markersize=1.5,
                                        markerfacecolor='#333333',
                                        markeredgecolor='none', alpha=0.4))
    for patch, s in zip(bp['boxes'], species):
        patch.set_facecolor(palette[s])
        patch.set_edgecolor('black')
    for median in bp['medians']:
        median.set_color('black')
    for s in species:
        exp = expected_ratios.get(s, np.nan)
        if np.isfinite(exp):
            ax_box.axhline(exp, color='#d62728', ls='--', lw=0.7, zorder=0)

    ax_scatter.set_xlabel('log₂(intensity)', fontsize=label_fontsize)
    ax_scatter.set_ylabel('log₂(ratio)', fontsize=label_fontsize)
    ax_scatter.legend(fontsize=legend_fontsize, frameon=False, loc='upper left',
                      markerscale=3)
    ax_density.set_xlabel('Density', fontsize=label_fontsize)
    ax_box.set_xticks(range(len(species)))
    ax_box.set_xticklabels([s[0] for s in species], fontsize=tick_fontsize)
    if ylim is not None:
        ax_scatter.set_ylim(*ylim)
    for ax in axes:
        ax.tick_params(labelsize=tick_fontsize)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in (ax_density, ax_box):
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    return fig, axes, stats, ratio_df
