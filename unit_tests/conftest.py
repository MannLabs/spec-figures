"""Shared pytest fixtures for the characterization suite.

These tests lock the *current* behaviour of the public API (REFACTOR_PLAN.md
§9) against the tracked PEAKS fixture before any module split, so the
reorganization cannot silently change results. Values here were captured from
the working library on 2026-07-15; a diff means behaviour changed.

The fixture is the two-condition PEAKS experiment used throughout the repo's
_check_* scripts: three 500SPD replicates (_A4/_A5/_A6) vs three 200SPD
replicates (_A10/_A11/_A12), loaded with library defaults
(peaks_pg_method='auto' -> MaxLFQ from the sibling proteins.csv).
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use('Agg')  # headless; no display needed for render tests

import pytest

import spec_analytics as core

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PEAKS = REPO_ROOT / 'Input_files' / 'PEAKS'
FASTA = REPO_ROOT / 'Input_files' / 'Human.fasta'

EXPERIMENTS = [
    {'path': str(PEAKS / '8600_500SPD_65vw_5ms' / 'lfq.dia.features.csv'),
     'file_tags': ['_A4', '_A5', '_A6'],
     'condition1': 'Zeno2', 'condition2': '500SPD'},
    {'path': str(PEAKS / '8600_200SPD_65vw_5ms' / 'lfq.dia.features.csv'),
     'file_tags': ['_A10', '_A11', '_A12'],
     'condition1': 'Zeno2', 'condition2': '200SPD'},
]


@pytest.fixture(scope='session')
def loaded():
    """(df, sample_info) from the canonical two-condition PEAKS experiment.

    Session-scoped and returned as copies-on-demand: tests must not mutate the
    shared frames. Loading is the expensive step, done once per session.

    The PEAKS fixture belongs to the analytics library's own repository and is not
    deposited with the figure code, so the characterization tests that need it skip
    rather than error when it is absent -- same rule as the FASTA fixture below.
    """
    missing = [e['path'] for e in EXPERIMENTS if not pathlib.Path(e['path']).exists()]
    if missing:
        pytest.skip(f'PEAKS fixture not present: {missing[0]}')
    df, sample_info = core.load_experiments(EXPERIMENTS)
    return df, sample_info


@pytest.fixture()
def df(loaded):
    return loaded[0]


@pytest.fixture()
def sample_info(loaded):
    return loaded[1]


@pytest.fixture(scope='session')
def agg(loaded):
    """Per-run summary from process_experiment(group_col='condition2')."""
    df, sample_info = loaded
    return core.process_experiment(df, sample_info, group_col='condition2')


@pytest.fixture(scope='session')
def protein_sequences():
    """{accession: sequence} from the human FASTA; skip the suite if absent."""
    if not FASTA.exists():
        pytest.skip(f'FASTA fixture not present: {FASTA}')
    return core.load_protein_sequences(str(FASTA))


@pytest.fixture(scope='session')
def protein_info(loaded, protein_sequences):
    """Per-(protein_group x condition2) coverage table for the loaded data."""
    df, sample_info = loaded
    return core.compute_protein_info(
        df, sample_info, protein_sequences, group_col='condition2')
