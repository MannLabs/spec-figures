"""Tests for targeted quantification — the parts that need no vendor raw file."""

import numpy as np
import pandas as pd
import pytest

from spec_analytics.raw import extraction as ex
from spec_analytics.raw import targeted as tg


def _gaussian(rt, center, fwhm, height, baseline=0.0):
    sigma = fwhm / 2.3548200450309493
    return baseline + height * np.exp(-0.5 * ((rt - center) / sigma) ** 2)


class TestBaselineSpec:
    def test_median_and_percentile_and_number(self):
        v = np.arange(101, dtype=float)
        assert ex._baseline_value(v, 'median') == 50.0
        assert ex._baseline_value(v, 'p5') == pytest.approx(5.0)
        assert ex._baseline_value(v, None) == 0.0
        assert ex._baseline_value(v, 7.5) == 7.5

    def test_unknown_spec_is_rejected(self):
        with pytest.raises(ValueError, match='baseline must be'):
            ex._baseline_value(np.arange(5.0), 'mean')


class TestPickPeak:
    def test_recovers_a_known_fwhm(self):
        rt = np.linspace(9.0, 11.0, 2001)
        peak = ex.pick_peak(rt, _gaussian(rt, 10.0, 0.10, 1e6), baseline=None)
        assert peak['apex_rt'] == pytest.approx(10.0, abs=1e-3)
        assert peak['fwhm'] == pytest.approx(0.10, abs=2e-3)

    def test_interpolation_recovers_width_a_coarse_grid_cannot(self):
        """The reason it exists: few scans quantise the FWHM to the cycle time."""
        rt = np.arange(9.5, 10.5, 0.02)          # 1.2 s cycle, ~5 points/peak
        y = _gaussian(rt, 10.0, 0.10, 1e6)
        coarse = ex.pick_peak(rt, y, baseline=None)['fwhm']
        fine = ex.pick_peak(rt, y, baseline=None, interpolate_points=1000)['fwhm']
        assert abs(fine - 0.10) < abs(coarse - 0.10)

    def test_min_sn_rejects_a_noise_trace(self):
        rt = np.linspace(9.0, 11.0, 401)
        weak = _gaussian(rt, 10.0, 0.1, 1.0, baseline=100.0)
        assert ex.pick_peak(rt, weak, baseline='p5', min_sn=3.0)['n_points'] == 0
        strong = _gaussian(rt, 10.0, 0.1, 1e5, baseline=100.0)
        assert ex.pick_peak(rt, strong, baseline='p5', min_sn=3.0)['n_points'] > 0

    def test_sn_is_reported(self):
        rt = np.linspace(9.0, 11.0, 401)
        y = _gaussian(rt, 10.0, 0.1, 900.0, baseline=100.0)
        assert ex.pick_peak(rt, y, baseline='p5')['sn'] == pytest.approx(10.0,
                                                                        rel=0.05)

    def test_empty_trace_is_not_a_crash(self):
        out = ex.pick_peak(np.array([]), np.array([]))
        assert out['n_points'] == 0 and out['area'] == 0.0


class TestPointsInWindow:
    def test_counts_acquired_scans_only(self):
        rt = np.arange(0.0, 1.0, 0.1)
        assert ex.points_in_window(rt, 0.5, 0.15) == 3      # 0.4, 0.5, 0.6

    def test_is_blind_to_interpolation(self):
        """Points across the peak is a property of the acquisition."""
        rt = np.arange(0.0, 1.0, 0.1)
        fine = np.linspace(0.0, 0.9, 1000)
        assert ex.points_in_window(rt, 0.5, 0.15) != ex.points_in_window(
            fine, 0.5, 0.15)


class TestIntegratePeak:
    def test_area_of_a_gaussian(self):
        rt = np.linspace(9.0, 11.0, 4001)
        height, fwhm = 1e6, 0.10
        y = _gaussian(rt, 10.0, fwhm, height)
        got = ex.integrate_peak(rt, y, 9.0, 11.0, baseline='none')['area']
        expected = height * fwhm / 2.3548200450309493 * np.sqrt(2 * np.pi)
        assert got == pytest.approx(expected, rel=1e-3)

    def test_min_boundary_baseline_removes_a_flat_background(self):
        rt = np.linspace(9.0, 11.0, 2001)
        clean = _gaussian(rt, 10.0, 0.1, 1e6)
        raised = clean + 5e4
        a = ex.integrate_peak(rt, clean, 9.5, 10.5, baseline='none')['area']
        b = ex.integrate_peak(rt, raised, 9.5, 10.5, baseline='min_boundary')['area']
        assert b == pytest.approx(a, rel=0.02)

    def test_window_is_honoured_exactly(self):
        rt = np.linspace(0.0, 10.0, 1001)
        out = ex.integrate_peak(rt, np.ones_like(rt), 2.0, 4.0, baseline='none')
        assert out['area'] == pytest.approx(2.0, rel=1e-6)

    def test_negative_integrand_is_clipped_not_subtracted(self):
        """A baseline above the signal means absent, not negative abundance."""
        rt = np.linspace(0.0, 1.0, 101)
        out = ex.integrate_peak(rt, np.ones_like(rt), 0.0, 1.0, baseline=5.0)
        assert out['area'] == 0.0

    def test_too_few_points_returns_zero_not_nan(self):
        out = ex.integrate_peak(np.array([1.0]), np.array([5.0]), 0.0, 2.0)
        assert out['area'] == 0.0 and out['n_points'] == 1


class TestFragmentFilters:
    def _frags(self):
        return pd.DataFrame({
            'target_idx': [0] * 4,
            'precursor_mz': [500.0] * 4,
            'ion_type': ['y', 'y', 'b', 'y'],
            'ion_number': [3, 5, 4, 2],
            'product_mz': [600.0, 700.0, 800.0, 400.0],
        })

    def test_keeps_only_c_terminal_ions_above_the_precursor(self):
        out = tg.y_ions_above_precursor(self._frags())
        assert sorted(out['product_mz']) == [600.0, 700.0]

    def test_margin_pushes_the_cut_up(self):
        out = tg.y_ions_above_precursor(self._frags(), margin=150.0)
        assert list(out['product_mz']) == [700.0]

    def test_top_n_keeps_the_strongest(self):
        f = self._frags()
        out = tg.top_n_fragments(f, [1, 9, 5, 3], n=2)
        assert sorted(out['product_mz']) == [700.0, 800.0]


class TestCalibrationFit:
    def _series(self, slope=3.0, intercept=0.0, cv=0.0, seed=0):
        rng = np.random.default_rng(seed)
        conc, area = [], []
        for c in (20, 200, 2000, 20000, 200000):
            for _ in range(4):
                conc.append(c)
                area.append((slope * c + intercept) * (1 + cv * rng.standard_normal()))
        return np.array(conc, float), np.array(area, float)

    def test_recovers_a_noise_free_line(self):
        fit = tg.calibration_fit(*self._series(slope=3.0, intercept=7.0))
        assert fit['slope'] == pytest.approx(3.0, rel=1e-6)
        assert fit['intercept'] == pytest.approx(7.0, abs=1e-6)
        assert fit['r_squared'] == pytest.approx(1.0, abs=1e-9)

    def test_weighting_protects_the_low_end(self):
        """1/x2 equalises relative residuals; unweighted lets the top point win."""
        conc, area = self._series(cv=0.05, seed=1)
        w = tg.calibration_fit(conc, area, weighting='1/x2')
        u = tg.calibration_fit(conc, area, weighting='none')
        lowest = conc == conc.min()
        rel = lambda f: abs(  # noqa: E731
            (area[lowest].mean() - f['intercept']) / f['slope'] - conc.min()
        ) / conc.min()
        assert rel(w) < rel(u)

    def test_accuracy_is_back_calculated_concentration(self):
        fit = tg.calibration_fit(*self._series())
        assert fit['levels']['accuracy_pct'].to_numpy() == pytest.approx(
            100.0, abs=1e-6)

    def test_lod_and_loq_scale_with_the_lowest_level_sd(self):
        fit = tg.calibration_fit(*self._series(cv=0.10, seed=2))
        assert fit['loq'] == pytest.approx(fit['lod'] * 10.0 / 3.3, rel=1e-9)
        assert fit['lod'] > 0

    def test_degenerate_and_short_inputs_are_rejected(self):
        with pytest.raises(ValueError, match='at least two'):
            tg.calibration_fit([1.0], [1.0])
        with pytest.raises(ValueError, match='degenerate'):
            tg.calibration_fit([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])

    def test_unknown_weighting_is_rejected(self):
        with pytest.raises(ValueError, match='weighting'):
            tg.calibration_fit([1.0, 2.0], [1.0, 2.0], weighting='1/x')

    def test_decoy_background_sets_a_measured_floor(self):
        conc, area = self._series(slope=3.0)
        # Background equivalent to ~100 amol of analyte; at 3x that the floor
        # should land near 300 amol.
        fit = tg.calibration_fit(conc, area, decoy_area=np.full(20, 300.0))
        assert fit['decoy_area_median'] == 300.0
        assert fit['lod_decoy'] == pytest.approx(300.0, rel=1e-6)

    def test_zero_decoy_background_gives_no_floor_rather_than_a_bogus_one(self):
        """Solving the line for a zero background returns -intercept/slope,
        which is a fact about the intercept, not about the background."""
        conc, area = self._series(slope=3.0, intercept=500.0)
        fit = tg.calibration_fit(conc, area, decoy_area=np.zeros(20))
        assert fit['decoy_area_median'] == 0.0
        assert np.isnan(fit['lod_decoy'])
        assert np.isfinite(fit['lod'])       # the SD-based limit still stands

    def test_no_decoys_supplied_leaves_the_fields_undefined(self):
        fit = tg.calibration_fit(*self._series())
        assert np.isnan(fit['lod_decoy'])
        assert np.isnan(fit['decoy_area_median'])


def test_activation_presets_are_distinct_ion_series():
    assert set(tg.FRAGMENT_TYPES['cid']) & set(tg.FRAGMENT_TYPES['ead']) == set()
    assert all(t[0] in 'by' for t in tg.FRAGMENT_TYPES['cid'])
    assert all(t[0] in 'cz' for t in tg.FRAGMENT_TYPES['ead'])
