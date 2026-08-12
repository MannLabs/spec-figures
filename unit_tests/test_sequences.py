"""Characterization: in-silico digestion + sequence property helpers."""

from __future__ import annotations

import math

import pytest

import spec_analytics as core

SEQ = 'MKWVTFISLLFLFSSAYSR'  # serum albumin signal peptide fragment


def test_digest_missed_cleavages():
    assert len(core.digest_protein(SEQ, max_missed_cleavages=0)) == 2
    assert len(core.digest_protein(SEQ, max_missed_cleavages=1)) == 3
    fully_cleaved = [p[2] for p in core.digest_protein(SEQ, max_missed_cleavages=0)]
    assert fully_cleaved == ['MK', 'WVTFISLLFLFSSAYSR']


def test_digest_empty():
    assert core.digest_protein('') == []


def test_count_missed_cleavages():
    assert core.count_missed_cleavages(SEQ) == 1
    assert core.count_missed_cleavages('WVTFISLLFLFSSAYSR') == 0


def test_gravy():
    assert core.gravy(SEQ) == pytest.approx(0.931579, abs=1e-6)
    assert math.isnan(core.gravy(''))


def test_theoretical_coverage():
    assert core.theoretical_coverage(SEQ) == pytest.approx(100.0, abs=1e-6)


def test_build_peptide_id():
    assert core.build_peptide_id('PEPTIDEK', 'Oxidation@M', '3') == 'PEP(Oxidation@M)TIDEK'
    assert core.build_peptide_id('PEPTIDEK', '', '') == 'PEPTIDEK'
