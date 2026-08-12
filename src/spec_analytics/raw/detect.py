"""Decoy-calibrated detection of a precursor list in a DIA raw file.

Answers one question: **is this precursor actually present in this run?** — for a
list of precursors that may never have been identified, so no search-engine output
exists to lean on.

Why not just look for a peak. Over a whole gradient any set of transitions has some
scan where their summed intensity is large, and that scan is almost always a
co-eluting interference rather than the peptide. Two design choices follow, and
they are the whole method:

* **Score by pattern, never by intensity.** At every scan the transition
  intensities are compared with the library's relative intensities by spectral
  angle. The position chosen is the best pattern match. Height is not used to
  select anything.
* **Match the selection bias with decoys.** Taking the best of several hundred
  scans inflates the score even for pure noise, so every precursor is scored twice:
  once on its real transitions and once on the same transitions shifted in m/z by a
  few Da. The decoys go through identical code over identical scans with the same
  take-the-best rule, so the inflation cancels and their score distribution gives a
  calibrated threshold at a chosen false-positive rate.

When an expected retention time is supplied a **threshold-free** readout comes with
it: the distance from the best-scoring position to that RT. Genuine signal piles up
near zero, chance interference spreads over the gradient, and the excess near zero
estimates the present fraction without any cutoff being chosen. Read that estimate
against a positive control scored the same way — in a DIA run with real
interference, even precursors known to be present land at their true RT well under
100 % of the time, so the raw near-zero fraction understates presence while the
ratio to the control does not.

Developed for a ZenoTOF/Orbitrap instrument comparison, where it recovered signal
for ~half of the precursors one instrument's engine had not reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = ['detect_precursors', 'pattern_scores']


def _spectral_angle_rows(mat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Spectral angle of each row of `mat` against `ref`. 1 = identical."""
    num = mat @ ref
    den = np.linalg.norm(mat, axis=1) * np.linalg.norm(ref)
    cos = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    return 1.0 - 2.0 * np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi


def _smooth_columns(mat: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return mat
    kern = np.ones(k) / k
    out = np.empty_like(mat, dtype=float)
    for c in range(mat.shape[1]):
        out[:, c] = np.convolve(mat[:, c], kern, mode='same')
    return out


def pattern_scores(rt, intensities, rel_intensity, *, smooth_scans=3,
                   min_channels=3):
    """Per-scan spectral angle between measured transitions and a library pattern.

    `intensities` is (n_scans, n_transitions) and `rel_intensity` the library
    pattern for those same transitions, in the same order. Scans with fewer than
    `min_channels` non-zero transitions are set to NaN: a pattern match on one or
    two channels is not evidence, and leaving them in lets a single spike win the
    take-the-best step.

    Returns the per-scan score array, aligned with `rt`.
    """
    mat = _smooth_columns(np.asarray(intensities, dtype=float), smooth_scans)
    ref = np.asarray(rel_intensity, dtype=float)
    sa = _spectral_angle_rows(mat, ref)
    sa[(mat > 0).sum(axis=1) < min_channels] = np.nan
    return sa


def detect_precursors(ms, targets, *, rt_col=None, rt_window=None,
                      decoy_offset_da=6.7, tol_ppm=15.0, smooth_scans=3,
                      min_channels=3, fdr=0.05, near_seconds=10.0,
                      verbose=True):
    """Decide whether each precursor in `targets` is present in `ms`.

    Parameters
      ms:          an object exposing the spectra to search. Either an alpharaw
                   ``MSData_Base`` (as returned by ``load_ms_data``) or anything
                   with the same ``spectrum_df`` / ``peak_df`` layout.
      targets:     long DataFrame, one row per transition, with columns
                   ``precursor_id``, ``precursor_mz``, ``product_mz`` and
                   ``rel_intensity`` (the library pattern; any positive scale).
                   Optionally an expected-RT column named by `rt_col`.
      rt_col:      column of `targets` holding the expected RT in minutes. When
                   given, `delta_rt_s` and the threshold-free readout are
                   produced; when None only the thresholded call is.
      rt_window:   ``(lo, hi)`` in minutes to restrict the search, or None for
                   every MS2 scan whose isolation window contains the precursor.
      decoy_offset_da:
                   m/z shift for the null transitions. Should be larger than the
                   extraction tolerance and not a plausible neutral loss.
      fdr:         decoy false-positive rate at which the threshold is set.
      near_seconds:
                   half-width of the "landed at the expected RT" window.

    Returns one row per precursor with the best real and decoy pattern scores,
    where each was found, the calibrated threshold, and the `detected` call.

    The `detected` fraction is a LOWER bound on presence, and the right
    denominator is the same statistic on precursors known to be present. Score a
    positive-control set alongside and quote the ratio.
    """
    spec = ms.spectrum_df
    is_ms2 = spec['ms_level'].to_numpy() == 2
    rt_all = spec['rt'].to_numpy()
    ilo = spec['isolation_lower_mz'].to_numpy()
    ihi = spec['isolation_upper_mz'].to_numpy()
    starts = spec['peak_start_idx'].to_numpy()
    stops = spec['peak_stop_idx'].to_numpy()
    pmz = ms.peak_df['mz'].to_numpy()
    pint = ms.peak_df['intensity'].to_numpy()

    need = {'precursor_id', 'precursor_mz', 'product_mz', 'rel_intensity'}
    missing = need - set(targets.columns)
    if missing:
        raise ValueError(f'targets is missing columns: {sorted(missing)}')

    def _xic(idx, mz_targets):
        out = np.zeros((idx.size, mz_targets.size), dtype=float)
        tol = mz_targets * tol_ppm * 1e-6
        lo_t, hi_t = mz_targets - tol, mz_targets + tol
        for r, i in enumerate(idx):
            a, b = starts[i], stops[i]
            if b <= a:
                continue
            m = pmz[a:b]
            csum = np.concatenate([[0.0], np.cumsum(pint[a:b], dtype=np.float64)])
            lo_i = np.searchsorted(m, lo_t, side='left')
            hi_i = np.searchsorted(m, hi_t, side='right')
            out[r] = csum[hi_i] - csum[lo_i]
        return out

    rows = []
    groups = list(targets.groupby('precursor_id', sort=False))
    for n, (pid, g) in enumerate(groups, 1):
        prec_mz = float(g['precursor_mz'].iloc[0])
        real = g['product_mz'].to_numpy(dtype=float)
        ref = g['rel_intensity'].to_numpy(dtype=float)
        keep = is_ms2 & (ilo <= prec_mz) & (ihi >= prec_mz)
        if rt_window is not None:
            keep &= (rt_all >= rt_window[0]) & (rt_all <= rt_window[1])
        idx = np.nonzero(keep)[0]
        idx = idx[np.argsort(rt_all[idx])]
        rec = {'precursor_id': pid, 'n_scans': int(idx.size)}
        if idx.size < 5 or real.size < min_channels:
            rows.append({**rec, 'score': np.nan, 'decoy_score': np.nan})
            continue

        rt = rt_all[idx]
        mat = _xic(idx, np.concatenate([real, real + decoy_offset_da]))
        k = real.size
        for tag, sl in (('', slice(0, k)), ('decoy_', slice(k, 2 * k))):
            sa = pattern_scores(rt, mat[:, sl], ref, smooth_scans=smooth_scans,
                                min_channels=min_channels)
            if not np.isfinite(sa).any():
                rec[f'{tag}score'] = np.nan
                continue
            b = int(np.nanargmax(sa))
            rec[f'{tag}score'] = float(sa[b])
            rec[f'{tag}rt'] = float(rt[b])
            rec[f'{tag}intensity'] = float(mat[b, sl].sum())
        if rt_col is not None and rt_col in g.columns:
            exp = float(g[rt_col].iloc[0])
            rec['rt_expected'] = exp
            if np.isfinite(rec.get('rt', np.nan)):
                rec['delta_rt_s'] = (rec['rt'] - exp) * 60.0
            if np.isfinite(rec.get('decoy_rt', np.nan)):
                rec['decoy_delta_rt_s'] = (rec['decoy_rt'] - exp) * 60.0
        rows.append(rec)
        if verbose and n % 100 == 0:
            print(f'  detect_precursors: {n}/{len(groups)}', flush=True)

    out = pd.DataFrame(rows)
    thr = float(np.nanquantile(out['decoy_score'], 1 - fdr)) \
        if out['decoy_score'].notna().any() else np.nan
    out['threshold'] = thr
    out['detected'] = out['score'] > thr
    if 'delta_rt_s' in out.columns:
        out['near_expected_rt'] = out['delta_rt_s'].abs() <= near_seconds
    if verbose:
        msg = (f'detect_precursors: {len(out):,} precursors, threshold '
               f'{thr:.3f} at {fdr:.0%} decoy FPR, '
               f'{out["detected"].mean():.1%} detected')
        if 'near_expected_rt' in out.columns:
            msg += (f', {out["near_expected_rt"].mean():.1%} landing within '
                    f'{near_seconds:.0f} s of the expected RT')
        print(msg)
    return out
