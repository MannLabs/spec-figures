"""Engine-agnostic experiment loading and dispatch: infer the engine, discover
runs, build sample_info, and concatenate per-file loader output into the
canonical long DataFrame. Extracted from _core.py (REFACTOR_PLAN.md step 3);
behaviour unchanged. The concrete loaders (io.peaks / io.diann) are imported
lazily inside load_experiments."""

from __future__ import annotations

import os
import re
import warnings
from typing import Iterable

import pandas as pd

from ..schema import validate_df, validate_sample_info


def _tag_matches(run_name: str, tag: str) -> bool:
    """
    True when `tag` appears in `run_name` immediately followed by either the
    end of the string or a non-alphanumeric character. Prevents tags like
    `_E1` from accidentally matching `_E264` substrings inside run names.
    """
    return re.search(re.escape(tag) + r'(?![A-Za-z0-9])', run_name) is not None


def detect_engine(path: str) -> str:
    """
    Infer search engine from a file path. Currently:
      *.parquet   -> 'diann'
      *.csv|*.tsv -> 'peaks'   (Spectronaut will be disambiguated later)
    """
    p = path.lower()
    if p.endswith('.parquet'):
        return 'diann'
    if p.endswith('.csv') or p.endswith('.tsv'):
        return 'peaks'
    raise ValueError(
        f'cannot infer engine from path {path!r}; pass engine=... explicitly'
    )


def _discover_runs(path: str, engine: str) -> list[str]:
    """Open `path` briefly and return the list of run names it contains."""
    if engine == 'diann':
        df = pd.read_parquet(path, columns=['Run'])
        return sorted(df['Run'].drop_duplicates().tolist())
    if engine == 'peaks':
        cols = list(pd.read_csv(path, nrows=0).columns)
        # Dispatch on which per-run column family the file actually has, rather
        # than testing all three patterns per column: features.csv carries an
        # aggregate 'Avg. Area' column that also matches the protein.csv
        # '<run> Area' pattern and would otherwise be reported as a run named
        # 'Avg.'.
        norm = [c for c in cols
                if c.endswith(' Normalized Area') and 'Group' not in c]
        if norm:                                    # features.csv
            return sorted(c[:-len(' Normalized Area')] for c in norm)
        pep = [c for c in cols if c.startswith('Area ') and 'Group' not in c]
        if pep:                                     # peptides.csv
            return sorted(c[len('Area '):] for c in pep)
        prot = [c for c in cols                     # protein.csv
                if c.endswith(' Area') and 'Group' not in c
                and c not in ('Avg. Area', 'Average Area')]
        return sorted(c[:-len(' Area')] for c in prot)
    raise ValueError(f'unknown engine: {engine!r}')


def sample_info_from_experiments(experiments: list[dict]) -> pd.DataFrame:
    """
    Build a `sample_info` DataFrame from a list of engine-agnostic experiment specs.

    Each experiment is a dict with keys:
      path:        file path to search-engine output (DIA-NN .parquet or PEAKS .csv)
      file_tags:   substring or list of substrings that match against run names
      condition1:  free-form first condition label  (e.g. instrument, sample type)
      condition2:  free-form second condition label (e.g. gradient, treatment)
      replicate:   (optional) int; auto-numbered within the experiment if absent
      batch:       (optional) defaults to `condition1`
      engine:      (optional) override for engine detection; normally auto-inferred
                   from the path extension (.parquet -> diann, .csv/.tsv -> peaks)

    `condition1` and `condition2` are deliberately free-form: pick whatever two
    axes describe your comparison (instrument x method, gradient x cell-type,
    treatment x dose, ...). Plot / process functions take these by name via
    `x_col`, `hue_col`, `group_col`.

    Opens each `path` to discover real run names, matches by file_tags, and
    emits one `sample_info` row per matched run.
    """
    rows = []
    for exp in experiments:
        engine = exp.get('engine') or detect_engine(exp['path'])
        runs = _discover_runs(exp['path'], engine)
        tags = exp['file_tags']
        if isinstance(tags, str):
            tags = [tags]
        batch = exp.get('batch', exp['condition1'])
        rep_override = exp.get('replicate')
        rep_counter = 0
        for tag in tags:
            matched = [r for r in runs if _tag_matches(r, tag)]
            if not matched:
                warnings.warn(f'file_tag {tag!r} matched no run in {exp["path"]}')
                continue
            if len(matched) > 1:
                warnings.warn(
                    f'file_tag {tag!r} matched multiple runs in {exp["path"]}: {matched}'
                )
            for run in matched:
                rep_counter += 1
                rows.append({
                    'run': run,
                    'file_path': exp['path'],
                    'condition1': exp['condition1'],
                    'condition2': exp['condition2'],
                    'replicate': rep_override if rep_override is not None else rep_counter,
                    'batch': batch,
                    'engine': engine,
                })
    si = pd.DataFrame(rows)
    if not si.empty:
        validate_sample_info(si)
    return si


def load_experiments(
    experiments: list[dict],
    sample_info: pd.DataFrame | None = None,
    *,
    diann_qvalue_filter: float | None = 0.01,
    diann_pg_method: str = 'auto',
    peaks_qvalue_filter: float | None = None,
    peaks_pg_method: str = 'auto',
    peaks_grouping: str = 'cc',
    directlfq_num_cores: int = 8,
    directlfq_use_inmemory: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Auto-dispatch loader for an engine-agnostic experiment list.

    For each experiment, infer the engine (from path or `experiment['engine']`),
    call the appropriate loader, and concatenate the results. Returns the
    canonical long DataFrame plus the sample_info actually used.

    Parameters:
      experiments:           list of engine-agnostic experiment dicts
      sample_info:           optional pre-built sample_info; built from
                             `experiments` when None
      diann_qvalue_filter:   q-value threshold for DIA-NN loads (default 0.01)
      diann_pg_method:       'auto' (default; resolves to 'maxlfq' for
                             DIA-NN since the parquet always carries
                             PG.MaxLFQ), 'maxlfq' (same thing, explicit),
                             or 'directlfq' (recompute via DirectLFQ for
                             uniform comparison with PEAKS).
      peaks_qvalue_filter:   q-value threshold for PEAKS loads (default None)
      peaks_pg_method:       'auto' (default; use 'maxlfq' per experiment
                             when proteins.csv exists, fall back to
                             'directlfq' with a warning otherwise — fast
                             native quant where available), 'maxlfq' (use
                             the MaxLFQ-style values from PEAKS' proteins
                             .csv; raises if the sibling proteins.csv is
                             missing), or 'directlfq' (always recompute
                             via DirectLFQ — slower, but uniform across
                             every experiment).
      peaks_grouping:        'cc' (default; PEAKS' connected-components rule —
                             matches PEAKS' own protein.csv) or 'signature'
                             (DIA-NN-style signature-based parsimony — splits
                             paralog families with unique peptides).
      directlfq_num_cores:   number of cores used by DirectLFQ for any load
                             that uses `pg_intensity_method='directlfq'`.
                             Default 8.
    """
    if sample_info is None:
        sample_info = sample_info_from_experiments(experiments)

    # Lazy import to avoid circular dependency.
    from . import diann as _diann
    from . import peaks as _peaks

    # DIA-NN's parquet always carries PG.MaxLFQ, so 'auto' resolves to 'maxlfq'
    # at the load-experiments level — diann.load_diann itself doesn't need to
    # know about 'auto'.
    diann_pg_method_resolved = (
        'maxlfq' if diann_pg_method == 'auto' else diann_pg_method
    )

    dfs = []
    peaks_resolved = []  # per-PEAKS-file resolved pg method (only populated
                         # when peaks_pg_method='auto').
    for path, group in sample_info.groupby('file_path', sort=False):
        engine = group['engine'].iloc[0]
        if engine == 'diann':
            d = _diann.load_diann(
                path, group,
                qvalue_filter=diann_qvalue_filter,
                pg_intensity_method=diann_pg_method_resolved,
                directlfq_num_cores=directlfq_num_cores,
                directlfq_use_inmemory=directlfq_use_inmemory,
            )
        elif engine == 'peaks':
            # protein.csv is a sibling of either features.csv or peptides.csv.
            for needle in ('features.csv', 'peptides.csv'):
                if needle in path:
                    protein_csv = path.replace(needle, 'proteins.csv')
                    if not os.path.exists(protein_csv):
                        protein_csv = None
                    break
            else:
                protein_csv = None
            d = _peaks.load_peaks(
                path, group,
                protein_csv=protein_csv,
                qvalue_filter=peaks_qvalue_filter,
                pg_intensity_method=peaks_pg_method,
                protein_grouping=peaks_grouping,
                directlfq_num_cores=directlfq_num_cores,
                directlfq_use_inmemory=directlfq_use_inmemory,
            )
            peaks_resolved.append((path, d.attrs.get('pg_intensity_method')))
        else:
            raise ValueError(f'unknown engine for {path}: {engine!r}')
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df.attrs['diann_pg_method'] = diann_pg_method
    df.attrs['diann_pg_method_resolved'] = diann_pg_method_resolved
    df.attrs['diann_qvalue_filter'] = diann_qvalue_filter
    df.attrs['peaks_qvalue_filter'] = peaks_qvalue_filter
    df.attrs['peaks_pg_method'] = peaks_pg_method
    # Per-PEAKS-file resolved choice. Useful when peaks_pg_method='auto' to
    # see which files fell back to DirectLFQ.
    df.attrs['peaks_pg_method_resolved'] = peaks_resolved
    df.attrs['peaks_grouping'] = peaks_grouping
    validate_df(df)
    return df, sample_info


def split_by_engine(
    df: pd.DataFrame,
    sample_info: pd.DataFrame,
) -> Iterable[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Yield (engine_name, df_subset, sample_info_subset) for each engine present.

    This is the canonical pattern for running the same analysis pipeline on
    each engine independently — search engines must NEVER be mixed within a
    single statistic or plot.

        for engine, df_e, si_e in core.split_by_engine(df, sample_info):
            agg = core.process_experiment(df_e, si_e, group_col='method')
            core.plot_boxplot_with_points(agg, x_col='method', y_col='protein_group')
    """
    for engine in sample_info['engine'].drop_duplicates():
        si_sub = sample_info[sample_info['engine'] == engine].reset_index(drop=True)
        df_sub = df[df['engine'] == engine].reset_index(drop=True)
        yield engine, df_sub, si_sub
