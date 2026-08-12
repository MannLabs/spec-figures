"""Characterization: filtering + reshape helpers."""

from __future__ import annotations

import pytest

import spec_analytics as core


def test_filter_qvalue(df):
    kept = core.filter_qvalue(df, 0.01)
    assert kept.shape == (287364, 18)
    assert kept['protein_group'].nunique() == 5871
    assert (kept['qvalue'] < 0.01).all()


def test_filter_qvalue_none_is_passthrough(df):
    assert core.filter_qvalue(df, None).shape == df.shape


def test_to_wide_default(df):
    wide = core.to_wide(df)
    assert wide.shape == (6, 5906)
    assert wide.index.name == 'run'
    assert wide.columns.name == 'protein_group'


def test_to_wide_precursor(df):
    wide = core.to_wide(df, value='precursor_intensity', columns='precursor_id')
    assert wide.shape == (6, 69645)


def test_filter_runs_roundtrip(df, sample_info):
    # Restricting sample_info to one condition drops the other condition's rows.
    si_500 = core.filter_sample_info(sample_info, condition2='500SPD')
    assert si_500.shape[0] == 3
    filtered = core.filter_runs(df, si_500)
    assert filtered['run'].nunique() == 3
    assert set(filtered['run']) == set(si_500['run'])


def test_filter_sample_info_bad_value_raises(sample_info):
    with pytest.raises(ValueError):
        core.filter_sample_info(sample_info, condition2='NOPE')


def test_filter_outlier_runs_keep_all(df, sample_info):
    # Cohorts (3 runs each) smaller than min_runs_for_fit -> GMM not fitted,
    # everything kept. Deterministic path; also exercises the warnings.warn
    # branch that a regression in step 4 had broken.
    dff, sif, summary = core.filter_outlier_runs(
        df, sample_info, cohort_col='condition2', min_runs_for_fit=4, plot=False)
    assert list(summary.columns) == [
        'cohort', 'n_total', 'n_kept', 'n_dropped', 'threshold', 'dropped_runs']
    assert len(summary) == 2
    assert int(summary['n_dropped'].sum()) == 0
    assert dff['run'].nunique() == 6
    assert sif['run'].nunique() == 6


def test_add_combined_group(df, sample_info):
    df_out, si_out = core.add_combined_group(
        df, sample_info, cols=['condition1', 'condition2'])
    assert 'condition_combined' in df_out.columns
    assert 'condition_combined' in si_out.columns
    assert set(df_out['condition_combined'].unique()) == {
        'Zeno2 / 500SPD', 'Zeno2 / 200SPD'}
