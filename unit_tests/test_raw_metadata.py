"""Tests for raw/metadata.py — the parts that need no vendor DLL.

The Clearcore2 round-trip cannot run in CI (Windows + pythonnet + a .wiff), so
what is pinned here is the logic that decides what a run *is* from its
isolation-window layout, and the name parsing that feeds it. Those are the
parts that were wrong once and would be wrong silently again.
"""

import warnings

import numpy as np
import pytest

from spec_analytics.raw import metadata as md


def _windows(start, stop, width):
    """Contiguously tiled windows over [start, stop) — a DIA method."""
    centres = np.arange(start + width / 2, stop, width)
    return np.full(centres.size, float(width)), centres


class TestClassify:
    """scan_mode reads MSExperimentInfo.IsSwath, set on the product-ion
    experiments of any DIA method and clear on a targeted one."""

    def test_no_fragment_experiments_is_ms1(self):
        assert md._classify([], 0) == 'ms1'

    def test_variable_window_dia(self):
        # ~100 SWATH windows, as pydiAID would lay them out.
        assert md._classify([True] * 100, 100) == 'dia'

    def test_scanning_dia_is_swath_with_many_windows(self):
        # A 0.4 Th sweep across 400-900 m/z: SWATH, ~1250 windows.
        n = 1256
        assert n >= md._SCANNING_MIN_WINDOWS
        assert md._classify([True] * n, n) == 'scanning_dia'

    def test_targeted_list_is_not_dia_despite_dia_like_spacing(self):
        # The regression this test exists for: the 6x5 standard's 30 targets
        # over 415-567 m/z sit about one window-width apart, so a layout-based
        # rule called them DIA. The declared flag does not care.
        assert md._classify([False] * 30, 30) == 'targeted'

    def test_isotope_envelope_of_one_peptide_is_targeted(self):
        # Six 1 Th windows over one precursor's isotopes tile perfectly.
        assert md._classify([False] * 6, 6) == 'targeted'

    def test_a_single_swath_window_still_reads_as_dia(self):
        assert md._classify([True], 1) == 'dia'

    def test_any_swath_product_experiment_makes_it_dia(self):
        # Mixed methods are rare, but a run carrying SWATH windows at all is
        # not a plain target list, and calling it targeted would hide them.
        assert md._classify([False] * 99 + [True], 100) == 'dia'


class TestCoverage:
    """window_coverage is a reported diagnostic — does the schema tile its
    range or leave gaps — not an input to scan_mode."""

    def test_tiled_windows_cover_their_span(self):
        widths, centres = _windows(400, 900, 2.0)
        assert md._coverage(widths, centres) == pytest.approx(1.0, abs=0.01)

    def test_sparse_targets_cover_little(self):
        widths = np.ones(4)
        centres = np.array([400.0, 500.0, 600.0, 700.0])
        assert md._coverage(widths, centres) == pytest.approx(4 / 301, abs=1e-4)

    def test_undefined_without_two_centres(self):
        assert np.isnan(md._coverage(np.array([1.0]), np.array([500.0])))

    def test_all_nan_widths_give_nan_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert np.isnan(md._coverage(np.array([np.nan, np.nan]),
                                         np.array([500.0, 600.0])))


class TestCentreFromName:
    def test_targeted_name_gives_the_precursor(self):
        assert md._centre_from_name('TOF PI of 496.7 (100 - 2000)') == 496.7

    def test_variable_window_name_gives_the_midpoint(self):
        # "380.4 to 394.4" is a window, and its centre is what tiles the range.
        assert md._centre_from_name(
            'TOF PI of 380.4 to 394.4 (130 - 2000)') == pytest.approx(387.4)

    def test_ms1_name_has_no_precursor(self):
        assert md._centre_from_name('TOF MS (400 - 1500)') is None

    def test_unparseable_name_returns_none_rather_than_raising(self):
        assert md._centre_from_name('TOF PI of everything (100 - 2000)') is None


class TestResolve:
    def test_directory_becomes_a_wiff_glob(self, tmp_path):
        (tmp_path / 'b.wiff').write_bytes(b'')
        (tmp_path / 'a.wiff').write_bytes(b'')
        (tmp_path / 'notes.txt').write_text('ignored')
        found = [p.rsplit('\\', 1)[-1].rsplit('/', 1)[-1]
                 for p in md._resolve(str(tmp_path))]
        assert found == ['a.wiff', 'b.wiff']

    def test_explicit_list_passes_through(self):
        assert md._resolve(['x.wiff', 'y.wiff']) == ['x.wiff', 'y.wiff']


def test_unsupported_extension_raises_with_a_useful_message():
    with pytest.raises(NotImplementedError, match=r'\.wiff'):
        md.read_acquisition_metadata('run.raw')
