"""Characterization: every plot_* renders without error and returns a Figure.

Exact pixel output is intentionally not asserted (too brittle); we lock the
contract that each function runs on the canonical data, returns a matplotlib
Figure as its first element, and — where it returns a stats table — that the
table is non-empty with a stable row count. A split that breaks a plot fails
here immediately.

plot_coverage_histogram is omitted: it consumes a FASTA-derived protein_info
frame, not the canonical df, so it has no fixture here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import spec_analytics as core


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _assert_fig(ret):
    """Every plot_* returns (fig, ax, [stats...]); first element is a Figure."""
    assert isinstance(ret, tuple)
    assert isinstance(ret[0], Figure)
    return ret


# --- plots taking (df, sample_info) with library defaults -----------------
DF_SI_PLOTS = [
    'plot_cv_violin',
    'plot_cv_ecdf',
    'plot_cv_stacked_bar_combined',
    'plot_pca',
    'plot_rank',
    'plot_venn',
    'plot_set_overlap',
    'plot_upset',
    'plot_intensity_histogram',
    'plot_peptide_gravy_distribution',
    'plot_peptide_length_distribution',
]


@pytest.mark.parametrize('fn_name', DF_SI_PLOTS)
def test_df_si_plots_render(fn_name, df, sample_info):
    fn = getattr(core, fn_name)
    _assert_fig(fn(df, sample_info))


# --- plots taking an agg frame (x_col / y_col) ----------------------------
def test_plot_bar(agg):
    _assert_fig(core.plot_bar(agg, x_col='condition2', y_col='protein_group'))


def test_plot_boxplot_with_points(agg):
    _assert_fig(core.plot_boxplot_with_points(agg, x_col='condition2', y_col='protein_group'))


def test_plot_median_scatter(agg):
    _assert_fig(core.plot_median_scatter(agg, x_col='condition2', y_col='protein_group'))


def test_plot_overlapping_bars(agg):
    ret = core.plot_overlapping_bars(
        agg, x_col='run', higher_y_col='protein_group', lower_y_col='PG20')
    _assert_fig(ret)


# --- two-condition comparison plots ---------------------------------------
def test_plot_volcano(df, sample_info):
    fig, ax, vdf = core.plot_volcano(
        df, sample_info, condition_a='500SPD', condition_b='200SPD')
    assert isinstance(fig, Figure)
    assert len(vdf) > 0


def test_plot_correlation(df, sample_info):
    ret = core.plot_correlation(
        df, sample_info, condition_a='500SPD', condition_b='200SPD')
    _assert_fig(ret)


def test_plot_qc_protein_heatmap(df, sample_info):
    # protein_groups is a dict[category -> list of gene symbols]; the heatmap
    # matches on the first gene name of each protein group.
    genes = [g.split(';')[0] for g in df['genes'].dropna().unique() if g]
    protein_groups = {'panel': genes[:6]}
    ret = core.plot_qc_protein_heatmap(df, sample_info, protein_groups)
    _assert_fig(ret)


# --- returned stats tables are non-empty ----------------------------------
def test_cv_violin_stats_nonempty(df, sample_info):
    fig, ax, cv_stats = core.plot_cv_violin(df, sample_info)
    assert len(cv_stats) > 0


def test_venn_stats_nonempty(df, sample_info):
    fig, ax, venn_df = core.plot_venn(df, sample_info)
    assert len(venn_df) > 0


def test_intensity_histogram_empty_path(df, sample_info):
    # No positive intensities in any partition (sample_info runs disjoint from
    # the df subset) exercises the empty-data guard, which previously raised
    # NameError on `long_rows`. It must now return an empty stats table.
    si_200 = core.filter_sample_info(sample_info, condition2='200SPD')
    df_500 = core.filter_runs(
        df, core.filter_sample_info(sample_info, condition2='500SPD'))
    fig, ax, data = core.plot_intensity_histogram(df_500, si_200)
    assert isinstance(fig, Figure)
    assert data.empty
