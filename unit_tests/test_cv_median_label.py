"""The median-CV label on plot_cv_violin is a percentage in both axis modes.

Regression: the label read the value in *plot* units and suffixed '%', so on
the default fractional axis (``as_percent=False``) a median CV of 0.066 was
annotated "0.1%" instead of "6.6%" — wrong by a factor of 100, and wrong in the
direction that looks like an outstandingly good result.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

import spec_analytics as core  # noqa: E402


@pytest.fixture
def df_and_sample_info():
    """Two conditions whose protein CVs differ enough to tell their labels apart."""
    rng = np.random.default_rng(0)
    rows, runs = [], []
    for cond, cv in [('tight', 0.05), ('loose', 0.20)]:
        for rep in range(4):
            run = f'{cond}_{rep}'
            runs.append({'run': run, 'condition2': cond, 'condition1': 'x',
                         'replicate': rep + 1, 'engine': 'peaks'})
            for p in range(300):
                base = 1e6 * (p + 1)
                rows.append({
                    'run': run,
                    'engine': 'peaks',
                    'protein_group': f'PG{p}',
                    'pg_intensity': base * (1 + cv * rng.standard_normal()),
                    'precursor_id': f'PEP{p}_2',
                    'precursor_intensity': base,
                    'peptide_id': f'PEP{p}',
                    'peptide_intensity': base,
                })
    return pd.DataFrame(rows), pd.DataFrame(runs)


def _labels(ax):
    return [t.get_text() for t in ax.texts if t.get_text().endswith('%')]


def _as_pct(labels):
    return sorted(float(t.rstrip('%')) for t in labels)


@pytest.mark.parametrize('loc', ['below', 'inline'])
def test_label_is_a_percentage_on_a_fractional_axis(df_and_sample_info, loc):
    df, sample_info = df_and_sample_info
    fig, ax, stats = core.plot_cv_violin(
        df, sample_info, level='protein', as_percent=False,
        median_label_loc=loc, group_order=['tight', 'loose'])
    labels = _as_pct(_labels(ax))
    assert len(labels) == 2
    expected = sorted(stats['median_CV'] * 100)
    assert labels == pytest.approx(expected, abs=0.05)
    # The point of the regression: nothing collapses to a sub-1% reading.
    assert min(labels) > 1.0
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize('loc', ['below', 'inline'])
def test_same_label_whether_or_not_the_axis_is_in_percent(df_and_sample_info, loc):
    """as_percent changes the axis, never the quoted number."""
    df, sample_info = df_and_sample_info
    out = {}
    for as_percent in (False, True):
        fig, ax, _ = core.plot_cv_violin(
            df, sample_info, level='protein', as_percent=as_percent,
            median_label_loc=loc, group_order=['tight', 'loose'])
        out[as_percent] = _as_pct(_labels(ax))
        matplotlib.pyplot.close(fig)
    assert out[False] == pytest.approx(out[True], abs=1e-6)


def test_label_tracks_the_data(df_and_sample_info):
    """The tighter condition must carry the smaller label."""
    df, sample_info = df_and_sample_info
    fig, ax, _ = core.plot_cv_violin(
        df, sample_info, level='protein', as_percent=False,
        median_label_loc='below', group_order=['tight', 'loose'])
    # ax.texts is emitted in group order, so the first label is 'tight'.
    labels = [float(t.get_text().rstrip('%')) for t in ax.texts
              if t.get_text().endswith('%')]
    assert labels[0] < labels[1]
    matplotlib.pyplot.close(fig)
