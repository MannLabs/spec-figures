"""Tests for `spec_analytics.raw.detect` — decoy-calibrated presence calls.

Built on a synthetic run rather than a fixture file, so the ground truth is known:
some precursors are spiked in as real co-eluting transition groups, others exist
only as target lists over pure background. The test is that the method separates
them, and that its two failure modes stay closed — a decoy must not outscore a
real peak, and a bright single-channel spike must not be called a detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spec_analytics.raw.detect import detect_precursors, pattern_scores


N_SCANS = 200
RT0, RT1 = 1.0, 6.0
PATTERN = np.array([1.0, 0.8, 0.55, 0.35, 0.2, 0.1])


class _FakeMS:
    """Minimal stand-in for an alpharaw MSData_Base: centroid peaks per scan."""

    def __init__(self, spectra, isolation=(400.0, 410.0)):
        rt = np.linspace(RT0, RT1, len(spectra))
        mzs, ints, starts, stops = [], [], [], []
        pos = 0
        for s in spectra:
            m = np.asarray(sorted(s.keys()), dtype=float)
            v = np.array([s[k] for k in m], dtype=float)
            mzs.append(m)
            ints.append(v)
            starts.append(pos)
            pos += m.size
            stops.append(pos)
        self.spectrum_df = pd.DataFrame({
            'rt': rt,
            'ms_level': np.full(len(spectra), 2),
            'isolation_lower_mz': np.full(len(spectra), isolation[0]),
            'isolation_upper_mz': np.full(len(spectra), isolation[1]),
            'peak_start_idx': np.array(starts),
            'peak_stop_idx': np.array(stops),
        })
        self.peak_df = pd.DataFrame({
            'mz': np.concatenate(mzs) if mzs else np.array([]),
            'intensity': np.concatenate(ints) if ints else np.array([]),
        })


def _build(frag_mz, *, apex_scan=None, height=1e5, noise=50.0, seed=0,
           spike_channel=None):
    """A run with flat background, optionally one real co-eluting peptide peak."""
    rng = np.random.default_rng(seed)
    rt = np.linspace(RT0, RT1, N_SCANS)
    spectra = []
    for i in range(N_SCANS):
        s = {}
        # background: a few random peaks per scan, including at the decoy m/z
        for mz in frag_mz:
            s[float(mz)] = float(abs(rng.normal(0, noise)))
            s[float(mz + 6.7)] = float(abs(rng.normal(0, noise)))
        if apex_scan is not None:
            w = np.exp(-0.5 * ((i - apex_scan) / 2.0) ** 2)
            for mz, p in zip(frag_mz, PATTERN):
                s[float(mz)] += float(height * p * w)
        if spike_channel is not None and i == N_SCANS // 3:
            s[float(frag_mz[spike_channel])] += 1e7
        spectra.append(s)
    return _FakeMS(spectra), rt


def _targets(pid, frag_mz, rt_expected=None):
    d = pd.DataFrame({
        'precursor_id': pid,
        'precursor_mz': 405.0,
        'product_mz': frag_mz,
        'rel_intensity': PATTERN,
    })
    if rt_expected is not None:
        d['rt_expected'] = rt_expected
    return d


FRAGS = np.array([500.1, 611.2, 722.3, 833.4, 944.5, 1055.6])


def test_real_peak_beats_its_decoy():
    ms, rt = _build(FRAGS, apex_scan=120)
    out = detect_precursors(ms, _targets('P1', FRAGS), verbose=False)
    r = out.iloc[0]
    assert r['score'] > r['decoy_score'], 'real transitions must outscore decoys'
    assert r['score'] > 0.9


def test_absent_precursor_scores_like_its_decoy():
    """No peptide spiked in: the real channels are background, like the decoys."""
    ms, _ = _build(FRAGS, apex_scan=None)
    out = detect_precursors(ms, _targets('P1', FRAGS), verbose=False)
    r = out.iloc[0]
    assert abs(r['score'] - r['decoy_score']) < 0.35


def test_threshold_separates_present_from_absent():
    present, absent = [], []
    for i in range(6):
        ms, _ = _build(FRAGS, apex_scan=60 + 10 * i, seed=i)
        present.append(detect_precursors(ms, _targets(f'p{i}', FRAGS),
                                         verbose=False))
        ms0, _ = _build(FRAGS, apex_scan=None, seed=100 + i)
        absent.append(detect_precursors(ms0, _targets(f'a{i}', FRAGS),
                                        verbose=False))
    p = pd.concat(present)['score'].to_numpy()
    a = pd.concat(absent)['score'].to_numpy()
    assert p.min() > a.max(), (
        f'present {p.min():.3f} must outscore absent {a.max():.3f}')


def test_expected_rt_gives_delta_and_landing():
    apex = 120
    ms, rt = _build(FRAGS, apex_scan=apex)
    out = detect_precursors(ms, _targets('P1', FRAGS, rt_expected=rt[apex]),
                            rt_col='rt_expected', verbose=False)
    r = out.iloc[0]
    assert abs(r['delta_rt_s']) < 10.0
    assert bool(r['near_expected_rt'])


def test_single_channel_spike_is_not_a_detection():
    """One enormous peak in one transition is the classic false positive."""
    ms, _ = _build(FRAGS, apex_scan=None, spike_channel=0)
    out = detect_precursors(ms, _targets('P1', FRAGS), min_channels=3,
                            verbose=False)
    assert out.iloc[0]['score'] < 0.9


def test_pattern_scores_is_maximal_on_an_exact_match():
    mat = np.tile(PATTERN, (5, 1))
    sa = pattern_scores(np.arange(5), mat, PATTERN, smooth_scans=1)
    assert np.nanmax(sa) == pytest.approx(1.0, abs=1e-9)


def test_missing_columns_raise():
    bad = pd.DataFrame({'precursor_id': ['x'], 'product_mz': [500.0]})
    ms, _ = _build(FRAGS, apex_scan=None)
    with pytest.raises(ValueError, match='missing columns'):
        detect_precursors(ms, bad, verbose=False)
