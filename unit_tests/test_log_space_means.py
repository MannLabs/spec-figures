"""Every plot drawn on a log intensity axis averages replicates IN LOG SPACE.

log2 of a linear mean is not the same statistic as the mean of log2: for
log-normal intensities the former sits ~sigma^2/2 higher, which lifts noisy
low-abundance entities and compresses the apparent dynamic range. The axis, the
y=x diagonal, the correlation coefficients and the t-statistics all live in log
space, so the location estimator has to as well.

These tests pin the convention with values whose two means differ by a wide,
hand-checkable margin.
"""

import pandas as pd
import pytest

import spec_analytics as core


# Two runs per condition. Protein P1 is deliberately skewed (2^10 and 2^20), so
# mean-of-log = 15 while log2-of-mean = log2(528384) = 19.011 — a 4-log2 gap.
INTENSITIES = {
    ('a', 'r1', 'P1'): 2.0 ** 10, ('a', 'r2', 'P1'): 2.0 ** 20,
    ('a', 'r1', 'P2'): 2.0 ** 12, ('a', 'r2', 'P2'): 2.0 ** 12,
    ('b', 'r3', 'P1'): 2.0 ** 14, ('b', 'r4', 'P1'): 2.0 ** 16,
    ('b', 'r3', 'P2'): 2.0 ** 13, ('b', 'r4', 'P2'): 2.0 ** 13,
}


@pytest.fixture
def skewed():
    rows = [{'run': run, 'engine': 'diann', 'protein_group': pg,
             'genes': pg, 'pg_intensity': value, 'condition2': cond}
            for (cond, run, pg), value in INTENSITIES.items()]
    df = pd.DataFrame(rows)
    sample_info = (df[['run', 'engine', 'condition2']].drop_duplicates()
                   .reset_index(drop=True))
    return df, sample_info


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


def test_plot_correlation_means_in_log_space(skewed):
    df, sample_info = skewed
    _fig, _ax, plot_df = core.plot_correlation(
        df, sample_info, level='protein', group_col='condition2',
        condition_a='a', condition_b='b')
    got = plot_df.set_index('protein_group')['log2_mean_a']
    # mean-of-log, not log2-of-mean (which would be 19.011 for P1).
    assert got['P1'] == pytest.approx(15.0)
    assert got['P2'] == pytest.approx(12.0)
    assert plot_df.set_index('protein_group')['log2_mean_b']['P1'] == \
        pytest.approx(15.0)




def test_plot_volcano_fold_change_is_mean_of_log_difference(skewed):
    """Welch branch only: the moderated branch computes log2_fc inside
    `_ebayes_moderated_p`, where it has always been a mean-of-log difference,
    and its variance prior needs far more features than this fixture has."""
    df, sample_info = skewed
    _fig, _ax, volcano_df = core.plot_volcano(
        df, sample_info, level='protein', group_col='condition2',
        condition_a='a', condition_b='b', min_valid_per_condition=2,
        method='welch')
    got = volcano_df.set_index('protein_group')['log2_fc']
    # P1: 15 - 15 = 0 in log space; log2 of the linear-mean ratio would be
    # 19.011 - 15.293 = +3.72, i.e. a spurious 13-fold change.
    assert got['P1'] == pytest.approx(0.0, abs=1e-9)
    # P2 has no within-condition spread, so both conventions agree: 12 - 13.
    assert got['P2'] == pytest.approx(-1.0)
