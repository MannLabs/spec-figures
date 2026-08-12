"""Tests for the annotated-spectrum panel and the spectrum-subset view."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')

import spec_analytics as core  # noqa: E402
from spec_analytics.raw import extraction as ex  # noqa: E402


class _MS:
    def __init__(self, spectrum_df, peak_df):
        self.spectrum_df = spectrum_df
        self.peak_df = peak_df


@pytest.fixture
def ms():
    """Four spectra alternating EAD (nce 12) and CID (nce 54)."""
    peaks = pd.DataFrame({'mz': np.arange(40, dtype=float) + 100.0,
                          'intensity': np.arange(40, dtype=float) + 1.0})
    spec = pd.DataFrame({
        'ms_level': [1, 2, 2, 2],
        'nce': [0.0, 12.0, 54.0, 12.0],
        'rt': [1.0, 1.1, 1.2, 2.5],
        'precursor_mz': [-1.0, 500.0, 500.0, 500.0],
        'peak_start_idx': [0, 10, 20, 30],
        'peak_stop_idx': [10, 20, 30, 40],
    })
    return _MS(spec, peaks)


class TestFilterSpectra:
    def test_selects_one_activation(self, ms):
        out = core.raw.filter_spectra(ms, nce=12.0)
        assert list(out.spectrum_df['rt']) == [1.1, 2.5]

    def test_accepts_several_nce_values(self, ms):
        out = core.raw.filter_spectra(ms, nce=(12.0, 54.0))
        assert len(out.spectrum_df) == 3

    def test_peak_indices_still_point_at_the_right_peaks(self, ms):
        """The whole point of a view: no re-indexing needed."""
        out = core.raw.filter_spectra(ms, nce=54.0)
        row = out.spectrum_df.iloc[0]
        got = out.peak_df['mz'].to_numpy()[
            int(row['peak_start_idx']):int(row['peak_stop_idx'])]
        assert got[0] == 120.0 and got.size == 10

    def test_peak_table_is_shared_not_copied(self, ms):
        assert core.raw.filter_spectra(ms, nce=12.0).peak_df is ms.peak_df

    def test_combines_with_ms_level_and_rt(self, ms):
        out = core.raw.filter_spectra(ms, ms_level=2, rt_range=(1.0, 1.5))
        assert list(out.spectrum_df['rt']) == [1.1, 1.2]


def _frags(product_mz, ion_types, numbers):
    return pd.DataFrame({
        'product_mz': product_mz, 'ion_type': ion_types,
        'ion_number': numbers,
        'fragment_charge': [1] * len(product_mz),
    })


class TestAnnotatedSpectrum:
    def test_matches_within_tolerance_and_misses_outside(self):
        mz = np.array([200.0, 300.0, 400.0])
        inten = np.array([10.0, 20.0, 30.0])
        # 300.002 is 6.7 ppm away (matches at 20 ppm); 401 is 2500 ppm (does not).
        frags = _frags([300.002, 401.0], ['b', 'y'], [2, 3])
        fig, ax, matched = core.plot_annotated_spectrum(
            mz, inten, frags, tol=20.0)
        assert list(matched['ion_type']) == ['b']
        assert matched['observed_mz'].iloc[0] == 300.0
        assert abs(matched['error_ppm'].iloc[0]) < 10
        matplotlib.pyplot.close(fig)

    def test_picks_the_most_intense_peak_within_tolerance(self):
        mz = np.array([299.999, 300.001])
        inten = np.array([5.0, 50.0])
        fig, _, matched = core.plot_annotated_spectrum(
            mz, inten, _frags([300.0], ['y'], [4]))
        assert matched['intensity'].iloc[0] == 50.0
        matplotlib.pyplot.close(fig)

    def test_mz_range_restricts_the_panel(self):
        mz = np.array([100.0, 500.0, 1900.0])
        inten = np.array([1.0, 2.0, 3.0])
        frags = _frags([100.0, 500.0, 1900.0], ['b', 'b', 'y'], [1, 2, 3])
        fig, _, matched = core.plot_annotated_spectrum(
            mz, inten, frags, mz_range=(200, 1000))
        assert list(matched['observed_mz']) == [500.0]
        matplotlib.pyplot.close(fig)

    def test_ladder_is_drawn_for_every_residue(self):
        seq = 'PEPTIDE'
        mz = np.array([300.0])
        fig, ax, _ = core.plot_annotated_spectrum(
            mz, np.array([10.0]), _frags([300.0], ['b'], [3]), sequence=seq)
        residues = [t.get_text() for t in ax.texts if len(t.get_text()) == 1]
        assert ''.join(residues) == seq
        matplotlib.pyplot.close(fig)

    def test_highlighted_site_is_coloured_differently(self):
        fig, ax, _ = core.plot_annotated_spectrum(
            np.array([300.0]), np.array([10.0]),
            _frags([300.0], ['c'], [2]), sequence='PEPS IDE'.replace(' ', ''),
            highlight_sites=[4])
        singles = [t for t in ax.texts if len(t.get_text()) == 1]
        assert singles[3].get_color() != singles[0].get_color()
        matplotlib.pyplot.close(fig)

    def test_empty_spectrum_in_range_is_an_error_not_an_empty_panel(self):
        with pytest.raises(ValueError, match='empty'):
            core.plot_annotated_spectrum(
                np.array([100.0]), np.array([1.0]),
                _frags([100.0], ['b'], [1]), mz_range=(500, 600))

    def test_n_and_c_terminal_series_get_different_colours(self):
        assert core.plotting.ION_COLORS['b'] != core.plotting.ION_COLORS['y']
        assert core.plotting.ION_COLORS['c'] != core.plotting.ION_COLORS['z']
