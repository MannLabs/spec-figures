"""Tests for the default technical-QC panel script (`technical_qc_analysis.py`).

The script is a repo-root entry point rather than part of the package, so it is
imported by path here. Only the pieces whose failure is *silent* are covered:
the panel drawing itself fails loudly, but a run-name collision across
conditions produced plausible-looking, wrong numbers.

The script ships with the analytics library, not with the SPEC figure code, so
this module skips itself when it is not on the path.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

tqc = pytest.importorskip(
    'technical_qc_analysis',
    reason='technical_qc_analysis.py is not part of the SPEC figure repository')


def _frames(rows):
    """(df, sample_info) skeletons with just the columns the helper touches."""
    df = pd.DataFrame(rows)
    sample_info = (df[['run', 'engine', 'condition2']]
                   .drop_duplicates().reset_index(drop=True))
    return df, sample_info


def test_disambiguate_runs_leaves_distinct_names_alone():
    df, si = _frames([
        {'run': 'r1', 'engine': 'peaks', 'condition2': 'A'},
        {'run': 'r2', 'engine': 'peaks', 'condition2': 'B'},
    ])
    out_df, out_si = tqc._disambiguate_runs(df, si)
    assert list(out_df['run']) == ['r1', 'r2']
    assert list(out_si['run']) == ['r1', 'r2']


def test_disambiguate_runs_tags_shared_run_names():
    """The same acquisition searched by two engines must stay two conditions."""
    df, si = _frames([
        {'run': 'r1', 'engine': 'peaks', 'condition2': 'PEAKS arm'},
        {'run': 'r1', 'engine': 'diann', 'condition2': 'DIA-NN arm'},
        {'run': 'r2', 'engine': 'peaks', 'condition2': 'other'},
    ])
    out_df, out_si = tqc._disambiguate_runs(df, si)
    assert list(out_df['run']) == ['r1 [PEAKS arm]', 'r1 [DIA-NN arm]', 'r2']
    # and the per-condition run sets, which every panel selects on, no longer
    # overlap
    sets = out_si.groupby('condition2')['run'].apply(set)
    assert not (sets['PEAKS arm'] & sets['DIA-NN arm'])


def test_disambiguate_runs_raises_when_engine_cannot_separate():
    df, si = _frames([
        {'run': 'r1', 'engine': 'peaks', 'condition2': 'A'},
        {'run': 'r1', 'engine': 'peaks', 'condition2': 'B'},
    ])
    with pytest.raises(ValueError, match='more than one condition'):
        tqc._disambiguate_runs(df, si)


def test_n_corrected_cv_rescales_toward_the_target_n():
    """A 4-replicate median CV understates an 8-replicate one on equal data."""
    assert tqc.n_corrected_cv(10.0, 4, n_to=4) == pytest.approx(10.0)
    up = tqc.n_corrected_cv(10.0, 4, n_to=8)
    assert up > 10.0 and up == pytest.approx(10.699, abs=1e-3)
    assert tqc.n_corrected_cv(up, 8, n_to=4) == pytest.approx(10.0)
