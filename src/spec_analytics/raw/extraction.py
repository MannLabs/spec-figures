"""Manual-mode raw-file extraction on top of alpharaw.

A light, engine-agnostic layer for targeted work on vendor raw files
(.raw / .wiff / .mzML):

  * load a raw file through alpharaw and cache it as HDF5 (fast re-open);
  * build an extracted-ion chromatogram (XIC) for one or more m/z targets
    within an optional retention-time window;
  * pick the XIC peak inside that window (apex, FWHM bounds, area).

The peak-picking is the same "manual" recipe used for intact-protein SEC-MS
(experiment E339) — half-maximum above a median baseline — and
works identically for a peptide precursor XIC.

alpharaw (and its vendor readers) is imported lazily inside ``load_ms_data``,
so importing this module — and using ``extract_xic`` / ``pick_peak`` on data
you already have in memory — costs nothing extra. alphabase (already a core
dependency) is imported lazily inside ``peptide_mz``.

Data model (alpharaw ``MSData_Base``): ``ms_data.spectrum_df`` has one row per
spectrum with ``rt`` (minutes), ``ms_level`` and ``peak_start_idx`` /
``peak_stop_idx``; ``ms_data.peak_df`` is the flat ``mz`` / ``intensity`` table
those indices slice into. Any object exposing those two frames works here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# reader_type -> module that registers it with alpharaw's ms_reader_provider
_READER_MODULE = {
    'thermo': 'alpharaw.thermo', 'thermo_raw': 'alpharaw.thermo',
    'sciex': 'alpharaw.sciex', 'sciex_wiff': 'alpharaw.sciex',
    'sciex_raw': 'alpharaw.sciex', 'mzml': 'alpharaw.mzml',
}
_EXT_TO_READER = {
    '.raw': 'thermo', '.wiff': 'sciex', '.mzml': 'mzml', '.mzml.gz': 'mzml',
}


def _mz_window(mz, mz_tol, mz_tol_unit):
    """(low, high) inclusive m/z bounds for one target."""
    if mz_tol_unit == 'ppm':
        half = mz * mz_tol * 1e-6
    elif mz_tol_unit in ('da', 'Da', 'th', 'Th'):
        half = mz_tol
    else:
        raise ValueError(f"mz_tol_unit must be 'ppm' or 'Da', got {mz_tol_unit!r}")
    return mz - half, mz + half


def load_ms_data(raw_path, *, cache=True, cache_dir=None, centroided=True,
                 reader_type=None, verbose=True):
    """Load a vendor raw file via alpharaw, caching it as HDF5.

    On first call the raw file is read and written to
    ``<cache_dir or raw dir>/<stem>.ms_data.hdf``; subsequent calls reload that
    HDF (much faster, and no vendor DLLs needed) as long as it is at least as
    new as the raw file.

    Parameters
      raw_path:     path to a .raw (Thermo) / .wiff (Sciex) / .mzML file, or an
                    existing .ms_data.hdf produced by a previous call.
      cache:        write/read the HDF cache (default True).
      cache_dir:    directory for the HDF cache; defaults to the raw file's dir.
      centroided:   centroid peaks on read (default True). Pass False to keep
                    the full profile (e.g. intact-protein native spectra).
      reader_type:  override the extension-based reader ('thermo' | 'sciex' |
                    'mzml'); normally inferred from the extension.

    Returns the alpharaw ``MSData_Base`` instance (``.spectrum_df`` /
    ``.peak_df`` populated), ready for ``extract_xic`` / ``extract_targets``.
    """
    import os
    import importlib

    ext = os.path.splitext(raw_path)[1].lower()
    is_hdf = ext in ('.hdf', '.hdf5') or raw_path.lower().endswith('.ms_data.hdf')

    # Validate the arguments before importing alpharaw: it is an optional extra,
    # and a bad extension should raise ValueError whether or not it is installed.
    if reader_type is None and not is_hdf:
        reader_type = _EXT_TO_READER.get(ext)
        if reader_type is None:
            raise ValueError(
                f'cannot infer reader from extension {ext!r}; pass '
                f'reader_type= one of {sorted(set(_READER_MODULE))}'
            )

    from alpharaw.ms_data_base import MSData_Base, ms_reader_provider

    # A pre-built HDF cache can be opened directly.
    if is_hdf:
        data = MSData_Base(centroided=centroided)
        data.load_hdf(raw_path)
        if verbose:
            print(f'[raw] loaded HDF {raw_path}')
        return data

    stem = os.path.basename(raw_path)
    stem = stem[:-len(ext)] if ext else stem
    cdir = cache_dir if cache_dir is not None else os.path.dirname(os.path.abspath(raw_path))
    cache_path = os.path.join(cdir, stem + '.ms_data.hdf')

    if cache and os.path.exists(cache_path) and \
            os.path.getmtime(cache_path) >= os.path.getmtime(raw_path):
        data = MSData_Base(centroided=centroided)
        data.load_hdf(cache_path)
        if verbose:
            print(f'[raw] loaded cache {cache_path}')
        return data

    # Importing the vendor module registers its reader with the provider.
    mod = importlib.import_module(_READER_MODULE[reader_type])
    if hasattr(mod, 'register_readers'):
        mod.register_readers()
    reader = ms_reader_provider.get_reader(reader_type, centroided=centroided)
    if reader is None:
        raise RuntimeError(
            f'alpharaw has no {reader_type!r} reader available (vendor '
            f'dependencies missing?)'
        )
    if verbose:
        print(f'[raw] reading {raw_path} ({reader_type})')
    reader.import_raw(raw_path)

    if cache:
        os.makedirs(cdir, exist_ok=True)
        reader.save_hdf(cache_path)
        if verbose:
            print(f'[raw] cached -> {cache_path}')
    return reader


def extract_xic(ms_data, mz, *, mz_tol=10.0, mz_tol_unit='ppm', rt_range=None,
                ms_level=1, precursor_mz=None):
    """Extracted-ion chromatogram for one or more m/z targets.

    Parameters
      ms_data:      object with ``.spectrum_df`` and ``.peak_df`` (an alpharaw
                    reader or anything with the same columns).
      mz:           a single m/z, or a sequence of m/z summed into one trace
                    (e.g. several isotopes of the same precursor).
      mz_tol:       half-window tolerance (default 10).
      mz_tol_unit:  'ppm' (default) or 'Da'.
      rt_range:     (min, max) retention-time window in minutes; None = whole run.
      ms_level:     MS level to draw from (default 1); None = any level.
      precursor_mz: for MS2 (DIA) fragment XICs — keep only spectra whose
                    isolation window (``isolation_lower_mz`` /
                    ``isolation_upper_mz``) contains this m/z. Ignored when the
                    spectra carry no isolation columns.

    Returns a DataFrame with columns ``rt`` (minutes) and ``intensity`` (summed
    peak intensity in the tolerance window per spectrum), sorted by rt.
    """
    spec = ms_data.spectrum_df
    keep = np.ones(len(spec), dtype=bool)
    if ms_level is not None and 'ms_level' in spec.columns:
        keep &= spec['ms_level'].to_numpy() == ms_level
    rt = spec['rt'].to_numpy()
    if rt_range is not None:
        lo, hi = rt_range
        keep &= (rt >= lo) & (rt <= hi)
    if precursor_mz is not None and 'isolation_lower_mz' in spec.columns:
        ilo = spec['isolation_lower_mz'].to_numpy()
        ihi = spec['isolation_upper_mz'].to_numpy()
        # NaN isolation (e.g. MS1 rows) compares False and is dropped, which is
        # correct: a precursor_mz filter only makes sense for isolated MS2.
        keep &= (ilo <= precursor_mz) & (ihi >= precursor_mz)

    mzs = np.atleast_1d(np.asarray(mz, dtype=float))
    windows = [_mz_window(float(m), mz_tol, mz_tol_unit) for m in mzs]

    peak_mz = ms_data.peak_df['mz'].to_numpy()
    peak_int = ms_data.peak_df['intensity'].to_numpy()
    starts = spec['peak_start_idx'].to_numpy()
    stops = spec['peak_stop_idx'].to_numpy()

    idxs = np.nonzero(keep)[0]
    out_rt = np.empty(len(idxs), dtype=float)
    out_int = np.zeros(len(idxs), dtype=float)
    for k, i in enumerate(idxs):
        out_rt[k] = rt[i]
        smz = peak_mz[starts[i]:stops[i]]
        sint = peak_int[starts[i]:stops[i]]
        total = 0.0
        for lo_mz, hi_mz in windows:
            sel = (smz >= lo_mz) & (smz <= hi_mz)
            if sel.any():
                total += float(sint[sel].sum())
        out_int[k] = total

    order = np.argsort(out_rt)
    return pd.DataFrame({'rt': out_rt[order], 'intensity': out_int[order]})


def sum_spectra(ms_data, *, rt_range=None, ms_level=2, precursor_mz=None,
                bin_tol=20.0, bin_tol_unit='ppm'):
    """Sum spectra over a retention-time window into one merged spectrum.

    The MS2 counterpart of an XIC: gather every spectrum at ``ms_level`` whose
    rt is in ``rt_range`` (and, for DIA, whose isolation window contains
    ``precursor_mz``), concatenate their centroids and merge peaks within
    ``bin_tol`` into a single summed spectrum. Summing over one LC-peak FWHM
    (from ``pick_peak``) gives a clean pseudo-MS2 for a precursor.

    Parameters
      ms_data:        object with ``.spectrum_df`` and ``.peak_df``.
      rt_range:       (min, max) minutes; None = whole run.
      ms_level:       spectrum level to sum (default 2); None = any.
      precursor_mz:   if given and the spectra carry ``isolation_lower_mz`` /
                      ``isolation_upper_mz``, keep only windows covering it
                      (the DIA window / DDA precursor for this peptide).
      bin_tol:        peak-merge tolerance (default 20).
      bin_tol_unit:   'ppm' (default) or 'Da'.

    Returns a DataFrame ``mz`` (intensity-weighted centre) / ``intensity``
    (summed), sorted by mz. Empty when no spectra/peaks match.
    """
    spec = ms_data.spectrum_df
    keep = np.ones(len(spec), dtype=bool)
    if ms_level is not None and 'ms_level' in spec.columns:
        keep &= spec['ms_level'].to_numpy() == ms_level
    if rt_range is not None:
        rt = spec['rt'].to_numpy()
        lo, hi = rt_range
        keep &= (rt >= lo) & (rt <= hi)
    if precursor_mz is not None and 'isolation_lower_mz' in spec.columns:
        ilo = spec['isolation_lower_mz'].to_numpy()
        ihi = spec['isolation_upper_mz'].to_numpy()
        keep &= (ilo <= precursor_mz) & (ihi >= precursor_mz)

    starts = spec['peak_start_idx'].to_numpy()
    stops = spec['peak_stop_idx'].to_numpy()
    peak_mz = ms_data.peak_df['mz'].to_numpy()
    peak_int = ms_data.peak_df['intensity'].to_numpy()

    chunks_mz, chunks_int = [], []
    for i in np.nonzero(keep)[0]:
        chunks_mz.append(peak_mz[starts[i]:stops[i]])
        chunks_int.append(peak_int[starts[i]:stops[i]])
    if not chunks_mz:
        return pd.DataFrame({'mz': [], 'intensity': []})
    mz = np.concatenate(chunks_mz)
    inten = np.concatenate(chunks_int)
    if mz.size == 0:
        return pd.DataFrame({'mz': [], 'intensity': []})

    order = np.argsort(mz, kind='mergesort')
    mz, inten = mz[order], inten[order]
    # A new cluster starts wherever the gap to the previous peak exceeds the
    # tolerance evaluated at that peak's m/z.
    if bin_tol_unit == 'ppm':
        thr = mz[1:] * bin_tol * 1e-6
    elif bin_tol_unit in ('da', 'Da', 'th', 'Th'):
        thr = np.full(mz.size - 1, bin_tol)
    else:
        raise ValueError(f"bin_tol_unit must be 'ppm' or 'Da', got {bin_tol_unit!r}")
    group = np.concatenate([[0], np.cumsum(np.diff(mz) > thr)])
    summed = np.bincount(group, weights=inten)
    wmz = np.bincount(group, weights=inten * mz) / summed
    return pd.DataFrame({'mz': wmz, 'intensity': summed})


def match_peaks(spec_mz, spec_intensity, target_mz, *, tol=20.0, tol_unit='ppm',
                reduce='max'):
    """Match a spectrum's peaks to target m/z values within a tolerance.

    For each target m/z, look for peaks in ``[target±tol]`` and reduce them to
    one value: ``'max'`` (default; tallest peak) or ``'sum'`` (total). Used to
    read observed fragment intensities off a summed spectrum at the theoretical
    (or predicted) fragment m/z, e.g. for a spectral-angle comparison.

    Parameters
      spec_mz, spec_intensity:  the spectrum (any order).
      target_mz:                scalar or array of m/z to look up.
      tol / tol_unit:           match window (default 20 ppm).
      reduce:                   'max' or 'sum' within each window.

    Returns ``(matched_intensity, matched_mz)`` arrays aligned to ``target_mz``;
    intensity 0 and m/z NaN where nothing matched.
    """
    smz = np.asarray(spec_mz, dtype=float)
    sint = np.asarray(spec_intensity, dtype=float)
    targets = np.atleast_1d(np.asarray(target_mz, dtype=float))
    order = np.argsort(smz)
    smz, sint = smz[order], sint[order]

    out_int = np.zeros(targets.size, dtype=float)
    out_mz = np.full(targets.size, np.nan, dtype=float)
    for i, t in enumerate(targets):
        lo, hi = _mz_window(float(t), tol, tol_unit)
        a = np.searchsorted(smz, lo, side='left')
        b = np.searchsorted(smz, hi, side='right')
        if b <= a:
            continue
        seg_int, seg_mz = sint[a:b], smz[a:b]
        if reduce == 'max':
            j = int(np.argmax(seg_int))
            out_int[i], out_mz[i] = seg_int[j], seg_mz[j]
        elif reduce == 'sum':
            s = float(seg_int.sum())
            out_int[i] = s
            out_mz[i] = float((seg_mz * seg_int).sum() / s) if s > 0 else np.nan
        else:
            raise ValueError(f"reduce must be 'max' or 'sum', got {reduce!r}")
    return out_int, out_mz


def _baseline_value(intensity, baseline):
    """Resolve a baseline spec against a trace. 'p<k>' is the k-th percentile."""
    if baseline is None:
        return 0.0
    if isinstance(baseline, str):
        if baseline == 'median':
            return float(np.median(intensity))
        if baseline.startswith('p'):
            return float(np.percentile(intensity, float(baseline[1:])))
        raise ValueError(
            "baseline must be 'median', 'p<percentile>', None or a number, "
            f'got {baseline!r}')
    return float(baseline)


def pick_peak(rt, intensity, *, rt_range=None, height_fraction=0.5,
              baseline='median', interpolate_points=None, min_sn=None):
    """Pick the dominant chromatographic peak by half-maximum over a baseline.

    Finds the apex (max intensity), sets a threshold at
    ``baseline + height_fraction * (apex - baseline)`` and walks outward from
    the apex to the points where the trace drops to that threshold — the FWHM
    window when ``height_fraction=0.5``. Area is the trapezoidal integral over
    that window. This mirrors the intact-protein LC-peak recipe and applies
    unchanged to a peptide precursor XIC.

    Parameters
      rt, intensity:    equal-length arrays (e.g. from ``extract_xic``).
      rt_range:         optional (min, max) to restrict picking.
      height_fraction:  fraction of apex height above baseline (0.5 = FWHM).
      baseline:         'median' (default), ``'p<k>'`` for the k-th percentile
                        (``'p5'`` suits a targeted trace, where the peak can be
                        most of the window and the median then sits on its
                        flank), None/0 for zero, or a float.
      interpolate_points:
                        resample onto this many evenly spaced points before
                        measuring. Worth setting for a **targeted** peak, which
                        may carry only 4-12 scans: read off the raw grid the
                        FWHM is quantised to the cycle time, so peaks of
                        genuinely different width report the same number.
      min_sn:           return the empty result unless ``apex / baseline``
                        reaches this. Only meaningful with a non-zero baseline.
                        Use it to keep noise out of a width or points-per-peak
                        summary, rather than filtering afterwards on a width
                        that was never trustworthy.

    Returns a dict: ``apex_rt, apex_intensity, left_rt, right_rt, fwhm, area,
    baseline, sn, n_points``. All-NaN (area/fwhm 0) when there is no positive
    signal, or when ``min_sn`` is not met.

    ``n_points`` counts points on the grid actually measured, so with
    ``interpolate_points`` it counts interpolated points — for scans across the
    peak use :func:`points_in_window` on the raw trace.
    """
    rt = np.asarray(rt, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if rt_range is not None:
        lo, hi = rt_range
        m = (rt >= lo) & (rt <= hi)
        rt, intensity = rt[m], intensity[m]

    empty = dict(apex_rt=np.nan, apex_intensity=np.nan, left_rt=np.nan,
                 right_rt=np.nan, fwhm=0.0, area=0.0, baseline=np.nan,
                 sn=np.nan, n_points=0)
    if intensity.size == 0 or not np.any(intensity > 0):
        return empty

    if interpolate_points and rt.size >= 2:
        fine_rt = np.linspace(rt.min(), rt.max(), int(interpolate_points))
        intensity = np.interp(fine_rt, rt, intensity)
        rt = fine_rt

    base = _baseline_value(intensity, baseline)
    apex = int(np.argmax(intensity))
    sn = float(intensity[apex] / base) if base > 0 else np.inf
    if min_sn is not None and not (sn >= min_sn):
        return empty
    thr = base + height_fraction * (intensity[apex] - base)

    lo = apex
    while lo > 0 and intensity[lo] > thr:
        lo -= 1
    hi = apex
    while hi < len(intensity) - 1 and intensity[hi] > thr:
        hi += 1

    # numpy 2.0 renamed trapz to trapezoid and 2.5 removed the old name. The
    # lookup has to be lazy: a default argument would evaluate the removed
    # attribute and raise even where the new name exists.
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    area = float(_trapz(intensity[lo:hi + 1], rt[lo:hi + 1]))
    return dict(
        apex_rt=float(rt[apex]), apex_intensity=float(intensity[apex]),
        left_rt=float(rt[lo]), right_rt=float(rt[hi]),
        fwhm=float(rt[hi] - rt[lo]), area=area, baseline=base, sn=sn,
        n_points=int(hi - lo + 1),
    )


class _SpectrumView:
    """A subset of an ms_data's spectra, sharing its peak table.

    ``peak_start_idx`` / ``peak_stop_idx`` index into ``peak_df``, so dropping
    rows from ``spectrum_df`` leaves the surviving rows pointing at exactly the
    same peaks. Nothing needs copying or re-indexing — the view is free however
    large the file.
    """

    def __init__(self, spectrum_df, peak_df):
        self.spectrum_df = spectrum_df
        self.peak_df = peak_df


def filter_spectra(ms_data, *, ms_level=None, nce=None, rt_range=None,
                   query=None):
    """Restrict an ms_data to a subset of its spectra, without copying peaks.

    The reason this exists: an instrument method can interleave fragmentation
    types on the same precursor — a Sciex MRM-HR run alternating EAD (NCE ~12)
    and CID (NCE ~54) scan by scan is one file containing two experiments. An
    XIC over all of them sums c/z and b/y evidence into one meaningless trace,
    so the activation has to be selected before extraction.

    :param ms_level: keep only this MS level.
    :param nce: a value or an iterable of values of the ``nce`` column. Matched
        with a tolerance of 0.5, since it is stored as a float.
    :param rt_range: ``(min, max)`` in minutes.
    :param query: a pandas query string, for anything else.
    :returns: a lightweight view exposing ``spectrum_df`` / ``peak_df``, usable
        anywhere an alpharaw reader is.
    """
    spec = ms_data.spectrum_df
    mask = np.ones(len(spec), dtype=bool)
    if ms_level is not None:
        mask &= spec['ms_level'].to_numpy() == ms_level
    if nce is not None:
        values = [nce] if np.isscalar(nce) else list(nce)
        nce_col = spec['nce'].to_numpy()
        hit = np.zeros(len(spec), dtype=bool)
        for v in values:
            hit |= np.abs(nce_col - float(v)) < 0.5
        mask &= hit
    if rt_range is not None:
        rt = spec['rt'].to_numpy()
        mask &= (rt >= rt_range[0]) & (rt <= rt_range[1])
    out = spec[mask]
    if query:
        out = out.query(query)
    return _SpectrumView(out, ms_data.peak_df)


def points_in_window(rt, center, half_width):
    """Number of acquired points within ``center +/- half_width``.

    Points across the peak is a property of the acquisition, so count it on the
    raw scan grid — never on an interpolated trace, which would report whatever
    resolution the interpolation used.
    """
    rt = np.asarray(rt, dtype=float)
    return int(np.sum((rt >= center - half_width) & (rt <= center + half_width)))


def integrate_peak(rt, intensity, rt_start, rt_end, *, baseline='min_boundary'):
    """Trapezoidal area between two retention times, over a baseline.

    Separate from :func:`pick_peak` because the integration window is *given*
    rather than found. In a dilution series the low levels have no pickable
    peak, and letting each level find its own window makes the window a
    function of the concentration — the one thing a calibration curve must not
    do. Anchor the window on a level where the peak is strong, then pass it
    here for every level.

    baseline:
      ``'min_boundary'`` (default) the lower of the two window edges, held flat.
        Robust against a sloping background, and it cannot eat into the peak
        the way a median would when the peak fills most of the window.
      ``'none'``  integrate from zero.
      ``'median'`` / ``'p<k>'`` / a float — as in :func:`pick_peak`.

    Returns ``area, baseline, n_points, rt_start, rt_end``. The integrand is
    clipped at zero: a negative area means the baseline sat above the signal,
    which is an absent peak, not negative abundance.
    """
    rt = np.asarray(rt, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    m = (rt >= rt_start) & (rt <= rt_end)
    rt_w, int_w = rt[m], intensity[m]
    if rt_w.size < 2:
        return dict(area=0.0, baseline=np.nan, n_points=int(rt_w.size),
                    rt_start=float(rt_start), rt_end=float(rt_end))

    if baseline == 'min_boundary':
        base = float(min(int_w[0], int_w[-1]))
    elif baseline == 'none':
        base = 0.0
    else:
        base = _baseline_value(int_w, baseline)

    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    area = float(_trapz(np.clip(int_w - base, 0, None), rt_w))
    return dict(area=area, baseline=base, n_points=int(rt_w.size),
                rt_start=float(rt_start), rt_end=float(rt_end))


def peak_from_fragments(ms_data, precursor_mz, fragment_mz, *, rt_range=None,
                        mz_tol=20.0, mz_tol_unit='ppm', top_n=6,
                        height_fraction=0.5, baseline='median'):
    """Locate a precursor's LC peak from the co-elution of its fragment ions.

    In DIA the MS1 precursor XIC is often not selective enough to place the
    elution peak — the fragments are. This builds an MS2-level XIC for each
    candidate ``fragment_mz`` (restricted to the DIA isolation window covering
    ``precursor_mz``), keeps the ``top_n`` most intense, sums them into one
    fragment chromatogram, and picks its peak.

    Pass fragment m/z that lie ABOVE the precursor m/z — cleaner, with less
    precursor-isotope and low-mass interference.

    Returns ``(peak, trace, used_mz)``: ``peak`` is the ``pick_peak`` dict on
    the summed fragment chromatogram (all-NaN if no fragment carries signal),
    ``trace`` is that chromatogram (``rt`` / ``intensity``), and ``used_mz`` the
    fragment m/z actually summed (most-intense first).
    """
    fragment_mz = np.atleast_1d(np.asarray(fragment_mz, dtype=float))
    traces, totals = [], []
    for fmz in fragment_mz:
        x = extract_xic(ms_data, float(fmz), mz_tol=mz_tol, mz_tol_unit=mz_tol_unit,
                        rt_range=rt_range, ms_level=2, precursor_mz=precursor_mz)
        traces.append(x)
        totals.append(float(x['intensity'].sum()))

    empty = pd.DataFrame({'rt': [], 'intensity': []})
    if not traces or all(len(t) == 0 for t in traces):
        return pick_peak(np.array([]), np.array([])), empty, np.array([])

    order = np.argsort(totals)[::-1]
    chosen = [i for i in order if totals[i] > 0][:top_n]
    rt = traces[order[0]]['rt'].to_numpy()
    if not chosen:
        return (pick_peak(rt, np.zeros(len(rt))),
                pd.DataFrame({'rt': rt, 'intensity': np.zeros(len(rt))}),
                np.array([]))
    summed = np.zeros(len(rt), dtype=float)
    for i in chosen:
        summed += traces[i]['intensity'].to_numpy()
    peak = pick_peak(rt, summed, height_fraction=height_fraction, baseline=baseline)
    return peak, pd.DataFrame({'rt': rt, 'intensity': summed}), fragment_mz[chosen]


def extract_targets(ms_data, targets, *, mz_tol=10.0, mz_tol_unit='ppm',
                    rt_tol=None, ms_level=1, height_fraction=0.5,
                    baseline='median'):
    """XIC + peak-pick a list of targets, one tidy row each.

    Parameters
      targets:  a DataFrame or a list of dicts. Each target supports:
                  ``mz``        (required) — float or list of m/z (summed);
                  ``name``      (optional) — label, defaults to the m/z;
                  ``rt``        (optional) — expected apex RT (minutes);
                  ``rt_range``  (optional) — explicit (min, max) window.
                If ``rt_range`` is absent but ``rt`` and ``rt_tol`` are given,
                the window is ``(rt - rt_tol, rt + rt_tol)``.
      rt_tol:   half-width (minutes) used to build a window around ``rt``.
      other params are passed through to extract_xic / pick_peak.

    Returns a DataFrame: one row per target with ``name, mz, expected_rt`` plus
    the ``pick_peak`` fields (apex_rt, apex_intensity, area, fwhm, ...).
    """
    if isinstance(targets, pd.DataFrame):
        targets = targets.to_dict('records')

    rows = []
    for t in targets:
        mz = t['mz']
        name = t.get('name', mz if np.isscalar(mz) else ';'.join(f'{m:g}' for m in mz))
        rr = t.get('rt_range')
        if rr is None and t.get('rt') is not None and rt_tol is not None:
            rr = (t['rt'] - rt_tol, t['rt'] + rt_tol)
        xic = extract_xic(ms_data, mz, mz_tol=mz_tol, mz_tol_unit=mz_tol_unit,
                          rt_range=rr, ms_level=ms_level)
        pk = pick_peak(xic['rt'].to_numpy(), xic['intensity'].to_numpy(),
                       height_fraction=height_fraction, baseline=baseline)
        rows.append({'name': name,
                     'mz': mz if np.isscalar(mz) else list(mz),
                     'expected_rt': t.get('rt'), **pk})
    return pd.DataFrame(rows)


def peptide_mz(sequence, charge, *, extra_mass=0.0):
    """Precursor m/z of a (stripped) peptide at a given charge.

    ``extra_mass`` adds the summed monoisotopic mass of any modifications
    (in Da) — pass 0 for the unmodified peptide. Uses alphabase's residue
    masses, so it is consistent with the rest of the package.
    """
    from alphabase.constants.aa import calc_AA_masses
    from alphabase.constants.atom import MASS_PROTON, MASS_H2O

    neutral = float(np.sum(calc_AA_masses(sequence))) + MASS_H2O + float(extra_mass)
    return (neutral + charge * MASS_PROTON) / charge
