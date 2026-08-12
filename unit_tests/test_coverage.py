"""Characterization: sequence-coverage pipeline + plot_coverage_histogram.

Requires the human FASTA fixture (Input_files/Human.fasta); the
`protein_sequences` fixture skips these tests if it is absent.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import spec_analytics as core

EXPECTED_PI_COLUMNS = [
    'protein_group', 'gene', 'group', 'representative_protein',
    'protein_length', 'n_peptides', 'coverage_pct',
]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_load_protein_sequences(protein_sequences):
    assert len(protein_sequences) == 20590
    # UniProt sp|ACC|NAME headers are parsed down to the bare accession.
    assert 'A0A0A0MS01' in protein_sequences
    assert protein_sequences['A0A0A0MS01'].isalpha()


def test_compute_protein_info_shape(protein_info):
    assert protein_info.shape == (10032, 7)
    assert list(protein_info.columns) == EXPECTED_PI_COLUMNS
    assert protein_info.groupby('group').size().to_dict() == {
        '200SPD': 5700, '500SPD': 4332}


def test_compute_protein_info_all_leaders_resolved(protein_info):
    # Every protein-group leader is present in the human FASTA.
    assert int(protein_info['coverage_pct'].isna().sum()) == 0
    assert (protein_info['coverage_pct'].between(0, 100)).all()


def test_compute_protein_info_median_coverage(protein_info):
    med = protein_info.groupby('group')['coverage_pct'].median()
    assert med['200SPD'] == pytest.approx(21.2121, abs=1e-3)
    assert med['500SPD'] == pytest.approx(18.3651, abs=1e-3)


def test_plot_coverage_histogram_renders(protein_info):
    fig, ax, source = core.plot_coverage_histogram(protein_info)
    assert isinstance(fig, Figure)
    assert source.shape == (10032, 3)
    assert list(source.columns) == ['protein_group', 'group', 'coverage_pct']


def test_plot_coverage_histogram_with_theoretical(protein_info, protein_sequences):
    pi = protein_info.copy()
    pi['theoretical'] = core.compute_theoretical_coverage(pi, protein_sequences)
    # Theoretical (tryptic) coverage far exceeds observed.
    assert pi['theoretical'].median() == pytest.approx(96.1214, abs=1e-2)
    fig, ax, source = core.plot_coverage_histogram(pi, theoretical_col='theoretical')
    assert isinstance(fig, Figure)
