"""Raw manual-mode extraction: XIC building, peak picking, peptide m/z.

Uses a synthetic in-memory MS object mimicking alpharaw's spectrum_df/peak_df,
so no vendor files or alpharaw import are needed. load_ms_data (which needs
alpharaw + vendor DLLs) is exercised only for its extension-inference error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spec_analytics.raw import (
    extract_xic, sum_spectra, pick_peak, peak_from_fragments, match_peaks,
    extract_targets, peptide_mz,
)


class _FakeMS:
    """Minimal stand-in exposing spectrum_df + peak_df like alpharaw."""
    def __init__(self, spectrum_df, peak_df):
        self.spectrum_df = spectrum_df
        self.peak_df = peak_df


TARGET = 500.25          # target m/z
APEX_SHAPE = [0.0, 100.0, 500.0, 300.0, 50.0, 0.0]  # target intensity per MS1 spectrum
RTS = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]


@pytest.fixture
def ms():
    """6 MS1 spectra (triangular target peak) interleaved with MS2 noise.

    Each MS1 spectrum carries three peaks: an off-target low peak, the target
    peak (intensity from APEX_SHAPE), and an off-target high peak. One MS2
    spectrum sits in the middle carrying a peak AT the target m/z that must be
    excluded when ms_level=1.
    """
    spec_rows, peak_mz, peak_int = [], [], []
    start = 0
    for rt, ta in zip(RTS, APEX_SHAPE):
        mzs = [400.10, TARGET, 700.90]
        ints = [7.0, ta, 3.0]
        peak_mz += mzs
        peak_int += ints
        spec_rows.append(dict(rt=rt, ms_level=1, peak_start_idx=start,
                              peak_stop_idx=start + len(mzs)))
        start += len(mzs)
    # an MS2 spectrum with a big peak at the target m/z (must be ignored at ms1)
    peak_mz += [TARGET]
    peak_int += [9999.0]
    spec_rows.append(dict(rt=10.25, ms_level=2, peak_start_idx=start,
                          peak_stop_idx=start + 1))

    spectrum_df = pd.DataFrame(spec_rows)
    peak_df = pd.DataFrame({'mz': peak_mz, 'intensity': peak_int})
    return _FakeMS(spectrum_df, peak_df)


def test_extract_xic_basic(ms):
    xic = extract_xic(ms, TARGET, mz_tol=10, mz_tol_unit='ppm')
    assert list(xic.columns) == ['rt', 'intensity']
    assert len(xic) == 6                       # only MS1 spectra
    assert xic['rt'].tolist() == RTS
    assert xic['intensity'].tolist() == APEX_SHAPE


def test_extract_xic_ignores_other_ms_level(ms):
    # The MS2 target peak (9999) must never appear at ms_level=1.
    xic = extract_xic(ms, TARGET, ms_level=1)
    assert xic['intensity'].max() == 500.0


def test_extract_xic_rt_range(ms):
    xic = extract_xic(ms, TARGET, rt_range=(10.15, 10.35))
    assert xic['rt'].tolist() == [10.2, 10.3]
    assert xic['intensity'].tolist() == [500.0, 300.0]


def test_extract_xic_tolerance_excludes_off_target(ms):
    # A tight window around a m/z with no peak yields an all-zero trace.
    xic = extract_xic(ms, 555.55, mz_tol=5, mz_tol_unit='ppm')
    assert xic['intensity'].sum() == 0.0


def test_extract_xic_multi_mz_sums(ms):
    # Summing the target and one off-target adds their per-spectrum intensities.
    xic = extract_xic(ms, [TARGET, 400.10], mz_tol=50, mz_tol_unit='ppm')
    assert xic['intensity'].tolist() == [a + 7.0 for a in APEX_SHAPE]


def test_pick_peak_apex_and_area(ms):
    xic = extract_xic(ms, TARGET)
    pk = pick_peak(xic['rt'].to_numpy(), xic['intensity'].to_numpy())
    assert pk['apex_rt'] == pytest.approx(10.2)
    assert pk['apex_intensity'] == pytest.approx(500.0)
    assert pk['left_rt'] < 10.2 < pk['right_rt']
    assert pk['fwhm'] > 0
    assert pk['area'] > 0


def test_pick_peak_empty_is_nan():
    pk = pick_peak(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    assert np.isnan(pk['apex_rt'])
    assert pk['area'] == 0.0
    assert pk['n_points'] == 0


def test_extract_targets_table(ms):
    targets = [
        {'name': 'peptideA', 'mz': TARGET, 'rt': 10.2},
        {'name': 'off', 'mz': 555.55},
    ]
    out = extract_targets(ms, targets, rt_tol=0.5)
    assert list(out['name']) == ['peptideA', 'off']
    a = out.set_index('name').loc['peptideA']
    assert a['apex_intensity'] == pytest.approx(500.0)
    assert a['expected_rt'] == 10.2
    assert np.isnan(out.set_index('name').loc['off', 'apex_rt'])


@pytest.fixture
def ms_ms2():
    """MS2 spectra in two isolation windows for sum_spectra tests.

    Two spectra in window [500,510] within the RT window carry the same two
    fragments (200.10, 300.20); a third spectrum sits in a different isolation
    window (must be excluded by precursor_mz); a fourth is out of the RT window.
    """
    rows, pmz, pint = [], [], []
    start = 0

    def add(rt, ilo, ihi, mzs, ints):
        nonlocal start
        rows.append(dict(rt=rt, ms_level=2, isolation_lower_mz=ilo,
                         isolation_upper_mz=ihi, peak_start_idx=start,
                         peak_stop_idx=start + len(mzs)))
        pmz.extend(mzs); pint.extend(ints)
        start += len(mzs)

    add(10.20, 500, 510, [200.100, 300.200], [10.0, 4.0])   # in window, in RT
    add(10.30, 500, 510, [200.101, 300.199], [6.0, 5.0])    # in window, in RT
    add(10.25, 600, 610, [200.100], [999.0])                # wrong window
    add(11.00, 500, 510, [200.100], [999.0])                # out of RT window

    return _FakeMS(pd.DataFrame(rows),
                   pd.DataFrame({'mz': pmz, 'intensity': pint}))


def test_sum_spectra_dia_window_and_rt(ms_ms2):
    spec = sum_spectra(ms_ms2, rt_range=(10.15, 10.35), ms_level=2,
                       precursor_mz=505.0, bin_tol=50, bin_tol_unit='ppm')
    assert list(spec.columns) == ['mz', 'intensity']
    assert len(spec) == 2                       # 200.1 and 300.2 merged bins
    d = dict(zip(spec['mz'].round(1), spec['intensity']))
    assert d[200.1] == pytest.approx(16.0)      # 10 + 6 summed across scans
    assert d[300.2] == pytest.approx(9.0)       # 4 + 5
    # binned m/z is the intensity-weighted centre of the two near-identical peaks
    frag = spec.loc[(spec['mz'] - 200.1).abs() < 0.01, 'mz'].iloc[0]
    assert 200.100 <= frag <= 200.101


def test_sum_spectra_excludes_wrong_window_and_rt(ms_ms2):
    # The 999-intensity peaks are in the wrong isolation window / RT and must
    # never leak in.
    spec = sum_spectra(ms_ms2, rt_range=(10.15, 10.35), precursor_mz=505.0)
    assert spec['intensity'].max() < 100


def test_match_peaks_max_and_miss():
    smz = np.array([200.100, 200.101, 300.200])
    sint = np.array([10.0, 6.0, 4.0])
    inten, mz = match_peaks(smz, sint, [200.1005, 250.0, 300.2],
                            tol=20, tol_unit='ppm')
    assert inten[0] == pytest.approx(10.0)      # tallest within window
    assert inten[1] == 0.0 and np.isnan(mz[1])  # no peak near 250
    assert inten[2] == pytest.approx(4.0)


def test_match_peaks_sum():
    smz = np.array([200.100, 200.101])
    sint = np.array([10.0, 6.0])
    inten, mz = match_peaks(smz, sint, 200.1005, tol=20, tol_unit='ppm', reduce='sum')
    assert inten[0] == pytest.approx(16.0)


@pytest.fixture
def ms_dia():
    """DIA MS2 scans in window [500,510] with two fragments (600.3, 700.4)
    eluting as a triangle across RT; a decoy scan in another window.
    """
    shape = [0.0, 50.0, 200.0, 120.0, 20.0, 0.0]
    rts = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]
    rows, pmz, pint = [], [], []
    start = 0
    for rt, a in zip(rts, shape):
        rows.append(dict(rt=rt, ms_level=2, isolation_lower_mz=500, isolation_upper_mz=510,
                         peak_start_idx=start, peak_stop_idx=start + 2))
        pmz += [600.300, 700.400]; pint += [a, a * 0.5]
        start += 2
    # decoy: different isolation window, big peak at 600.3 (must be excluded)
    rows.append(dict(rt=10.2, ms_level=2, isolation_lower_mz=600, isolation_upper_mz=610,
                     peak_start_idx=start, peak_stop_idx=start + 1))
    pmz += [600.300]; pint += [9999.0]
    return _FakeMS(pd.DataFrame(rows), pd.DataFrame({'mz': pmz, 'intensity': pint}))


def test_extract_xic_ms2_isolation_filter(ms_dia):
    # precursor_mz=505 selects the [500,510] window only; the decoy (window
    # [600,610]) is excluded even though it has a 600.3 peak.
    xic = extract_xic(ms_dia, 600.300, ms_level=2, precursor_mz=505.0, mz_tol=50)
    assert xic['intensity'].tolist() == [0.0, 50.0, 200.0, 120.0, 20.0, 0.0]


def test_peak_from_fragments(ms_dia):
    peak, trace, used = peak_from_fragments(
        ms_dia, precursor_mz=505.0, fragment_mz=[600.300, 700.400],
        rt_range=(9.9, 10.6), mz_tol=50)
    assert peak['apex_rt'] == pytest.approx(10.2)   # triangle apex
    assert peak['fwhm'] > 0
    assert len(used) == 2                            # both fragments carried signal
    # summed trace apex = 200 + 100 (600.3 + 700.4 at apex)
    assert trace['intensity'].max() == pytest.approx(300.0)


def test_peak_from_fragments_no_signal(ms_dia):
    # fragment m/z absent from the window -> all-NaN peak
    peak, trace, used = peak_from_fragments(
        ms_dia, precursor_mz=505.0, fragment_mz=[999.999], rt_range=(9.9, 10.6))
    assert np.isnan(peak['apex_rt'])
    assert len(used) == 0


def test_peptide_mz_matches_alphabase():
    from alphabase.constants.aa import calc_AA_masses
    from alphabase.constants.atom import MASS_PROTON, MASS_H2O
    seq, z = 'PEPTIDEK', 2
    expected = (float(np.sum(calc_AA_masses(seq))) + MASS_H2O + z * MASS_PROTON) / z
    assert peptide_mz(seq, z) == pytest.approx(expected, abs=1e-6)
    # higher charge -> lower m/z
    assert peptide_mz(seq, 3) < peptide_mz(seq, 2)


def test_load_ms_data_bad_extension():
    with pytest.raises(ValueError):
        # .txt has no reader mapping and no explicit reader_type
        __import__('spec_analytics.raw', fromlist=['load_ms_data'])
        from spec_analytics.raw import load_ms_data
        load_ms_data('nonexistent.txt', cache=False)
