"""Acquisition metadata from a vendor raw file, without decoding any peaks.

Answers "what did this run actually do" — method name, instrument, acquisition
timestamp, isolation-window layout, and whether the method was DIA, scanning
DIA or targeted — at roughly **0.2 s per file**, against minutes for a full
``import_raw``. The whole cost of loading a raw file is decoding the spectra;
the method description hangs off the sample and experiment headers and needs
none of it.

Two things make this worth having as its own entry point rather than a note in
a notebook:

* **Sorting runs by what they are.** A folder of raw files is usually a mix of
  methods, and the filename rarely says which is which. ``survey_raw_files``
  turns a folder into a table you can group by — ``method``, ``scan_mode``,
  ``isolation_width_median`` — in about as long as listing it.
* **The acquisition timestamp is authoritative.** ``AcquisitionDateTime`` is
  written by the instrument, unlike an mtime, which a file copy resets. Use
  ``acquisition_times`` to build the ``acq_order`` factor rather than ranking
  mtimes (see ``technical_qc_analysis._add_acquisition_order``).

**alpharaw's ``isolation_lower_mz`` / ``isolation_upper_mz`` are wrong for
targeted Sciex data.** ``pysciexwifffilereader`` only reads the real window
when ``Details.IsSwath`` is set, and otherwise hardcodes 3.0 Th — so an
MRM-HR run acquired at 0.5 Th unit resolution reports 3 Th in
``spectrum_df``. The true width lives on ``MassRangeInfo[k].IsolationWindow``,
which is what this module reads. Trust ``isolation_width_median`` from here
over anything derived from ``spectrum_df`` on non-SWATH data.

Sciex ``.wiff`` only for now — it is the format whose Python readers do not
already expose this. ``read_acquisition_metadata`` raises a clear
``NotImplementedError`` for anything else rather than guessing.
"""

from __future__ import annotations

import glob as _glob
import os

import numpy as np
import pandas as pd

__all__ = [
    'read_acquisition_metadata',
    'survey_raw_files',
    'acquisition_times',
]

#: Below this many isolation windows a SWATH run is called 'dia' rather than
#: 'scanning_dia'. This is the one part of ``scan_mode`` that is a judgement
#: rather than a declared fact — both are SWATH as far as the file is
#: concerned. A scanning quadrupole emits one window per bin across the whole
#: precursor range (~1250 for 400-900 m/z at 0.4 Th), while a variable-window
#: method is designed around ~100 windows; nothing real sits between.
_SCANNING_MIN_WINDOWS = 200


def _open_wiff(path):
    """Open a wiff batch and return (reader, sample, mass_spectrometer_sample)."""
    from alpharaw.raw_access.pysciexwifffilereader import WillFileReader

    reader = WillFileReader(path)
    sample = reader._wiff_file.GetSample(0)
    return reader, sample, sample.MassSpectrometerSample


def _experiment_rows(ms_sample):
    """One dict per MS experiment (the method's scan definitions)."""
    rows = []
    for i in range(ms_sample.ExperimentCount):
        details = ms_sample.GetMSExperiment(i).Details
        mass_ranges = details.MassRangeInfo
        first = mass_ranges[0] if len(mass_ranges) else None
        rows.append({
            'experiment': i,
            'experiment_type': str(details.ExperimentType),
            'name': str(details.Name),
            'n_scans': int(details.NumberOfScans),
            'start_mz': float(details.StartMass),
            'end_mz': float(details.EndMass),
            'polarity': str(details.Polarity),
            'resolution': float(details.DefaultResolution),
            'is_swath': bool(details.IsSwath),
            # IsolationWindow only exists on fragment-based ranges; an MS1
            # full-scan range has no window and correctly yields NaN.
            'isolation_width': float(getattr(first, 'IsolationWindow', np.nan))
            if first is not None else np.nan,
            'dwell_time_ms': float(getattr(first, 'DwellTime', np.nan))
            if first is not None else np.nan,
        })
    return rows


def _span(widths, centres):
    """Precursor range the isolation windows cover end to end, in Th."""
    return float(centres.max() - centres.min()) + float(np.nanmedian(widths))


def _coverage(widths, centres):
    """Summed window width over the spanned precursor range. ~1 when tiled.

    Reported as a diagnostic, not used to classify: it says whether the windows
    tile their range contiguously or leave gaps, which is worth seeing for a
    DIA method but does not identify one. Gaps in a DIA schema are a method
    bug; a targeted list has gaps by design.
    """
    # np.nanmedian warns on an all-NaN slice, so screen that out first — an
    # experiment type with no IsolationWindow at all is a normal input here.
    if widths.size == 0 or centres.size < 2 or np.isnan(widths).all():
        return np.nan
    span = _span(widths, centres)
    return float(np.nansum(widths)) / span if span > 0 else np.nan


def _classify(swath_flags, n_windows):
    """Scan mode, from what the method declares rather than from its layout.

    ``MSExperimentInfo.IsSwath`` is set on the *product-ion* experiments of a
    DIA method — both variable-window SWATH and the ZT Scan sweep — and clear
    on a targeted MRM-HR / PRM method. Note it is the product experiments that
    carry it; the survey TOF MS scan reads False in every method, which is what
    makes this look inconclusive if you only inspect experiment 0.

    Reading the flag matters because the layout does not always give the answer
    away: the 6x5 standard's 30 targets sit about one window-width apart over
    150 Th, and one peptide's isotope envelope tiles perfectly over 6 Th, so
    both mimic DIA's window spacing while being targeted runs.
    """
    if n_windows == 0:
        return 'ms1'
    if not any(swath_flags):
        return 'targeted'
    return 'scanning_dia' if n_windows >= _SCANNING_MIN_WINDOWS else 'dia'


def read_acquisition_metadata(path, *, with_experiments=False):
    """Method and instrument metadata for one raw file, without reading peaks.

    :param str path: path to a Sciex ``.wiff``.
    :param bool with_experiments: also return the per-experiment table under
        the ``'experiments'`` key (one row per scan definition in the method).
    :returns: a dict of scalars — see ``survey_raw_files`` for the columns.

    The precursor centre of a product-ion experiment is parsed from its display
    name (``"TOF PI of 496.7 (100 - 2000)"``), the one place Clearcore2 exposes
    it without touching a spectrum. Names that do not parse are skipped, so a
    method the parser does not recognise degrades to fewer windows counted
    rather than to wrong numbers.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext != '.wiff':
        raise NotImplementedError(
            f'read_acquisition_metadata supports Sciex .wiff only, got {ext!r}. '
            'Thermo .raw and mzML expose their method metadata through their '
            'own readers; add a branch here when one is needed.')

    reader, sample, ms_sample = _open_wiff(path)
    try:
        details = sample.Details
        exps = _experiment_rows(ms_sample)

        first = ms_sample.GetMSExperiment(0)
        n_cycles = int(first.Details.NumberOfScans)
        rt_start = float(first.GetRTFromExperimentCycle(0))
        rt_end = float(first.GetRTFromExperimentCycle(max(n_cycles - 1, 0)))

        products = [e for e in exps if e['experiment_type'] != 'MS']
        widths = np.array([e['isolation_width'] for e in products], dtype=float)
        centres = np.array([c for c in (_centre_from_name(e['name'])
                                        for e in products) if c is not None])

        meta = {
            'file': os.path.basename(path),
            'path': os.path.abspath(path),
            'sample_name': str(details.SampleName),
            'vial': str(details.Vial),
            'method': os.path.basename(str(details.AcquisitionMethodName)),
            'acquired': pd.Timestamp(details.AcquisitionDateTime.ToString('O')),
            'instrument': str(details.InstrumentName),
            'serial': str(details.InstrumentSerialNumber),
            'software': str(details.SoftwareVersion),
            'batch': str(details.BatchName),
            'operator': str(details.UserName),
            'n_experiments': len(exps),
            'n_windows': len(products),
            'n_cycles': n_cycles,
            'rt_start_min': rt_start,
            'rt_end_min': rt_end,
            # One cycle is one pass through every experiment, so the MS1
            # experiment's scan-to-scan spacing is the cycle time.
            'cycle_time_s': ((rt_end - rt_start) / (n_cycles - 1) * 60.0
                             if n_cycles > 1 else np.nan),
            'ms1_start_mz': exps[0]['start_mz'] if exps else np.nan,
            'ms1_end_mz': exps[0]['end_mz'] if exps else np.nan,
            'polarity': exps[0]['polarity'] if exps else '',
            'dwell_time_ms': float(np.nanmedian(
                [e['dwell_time_ms'] for e in exps])) if exps else np.nan,
            'isolation_width_median': (float(np.nanmedian(widths))
                                       if widths.size else np.nan),
            'isolation_width_min': (float(np.nanmin(widths))
                                    if widths.size else np.nan),
            'isolation_width_max': (float(np.nanmax(widths))
                                    if widths.size else np.nan),
            'precursor_min_mz': float(centres.min()) if centres.size else np.nan,
            'precursor_max_mz': float(centres.max()) if centres.size else np.nan,
            'is_swath': bool(any(e['is_swath'] for e in products)),
            'window_coverage': _coverage(widths, centres),
            'scan_mode': _classify([e['is_swath'] for e in products],
                                   len(products)),
        }
        if with_experiments:
            meta['experiments'] = pd.DataFrame(exps)
        return meta
    finally:
        reader.close()


def _centre_from_name(name):
    """Precursor centre m/z from a product-ion experiment's display name.

    ``"TOF PI of 496.7 (100 - 2000)"`` -> 496.7, and the variable-window form
    ``"TOF PI of 380.4 to 394.4 (130 - 2000)"`` -> the midpoint 387.4.
    Returns None when the name does not carry a precursor.
    """
    marker = ' of '
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1].split('(')[0].strip()
    parts = tail.split(' to ')
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    return float(np.mean(values))


def survey_raw_files(paths, *, verbose=True, errors='warn'):
    """Metadata table for many raw files — one row per run.

    :param paths: a list of paths, a directory, or a glob pattern.
    :param str errors: ``'warn'`` (default) records the failure in an ``error``
        column and keeps going; ``'raise'`` propagates.
    :returns: DataFrame sorted by acquisition time, so the row order *is* the
        acquisition order.

    Group by ``method`` or ``scan_mode`` to sort a mixed folder into its
    conditions before loading a single spectrum.
    """
    paths = _resolve(paths)
    rows = []
    for i, path in enumerate(paths, 1):
        try:
            rows.append(read_acquisition_metadata(path))
        except Exception as exc:                                # noqa: BLE001
            if errors == 'raise':
                raise
            rows.append({'file': os.path.basename(path),
                         'path': os.path.abspath(path),
                         'error': f'{type(exc).__name__}: {exc}'})
        if verbose:
            print(f'  survey_raw_files: {i}/{len(paths)} {rows[-1]["file"]}',
                  flush=True)
    df = pd.DataFrame(rows)
    if 'acquired' in df.columns:
        df = df.sort_values('acquired', kind='stable').reset_index(drop=True)
    return df


def acquisition_times(paths, *, verbose=False):
    """Instrument-written acquisition timestamp per run, keyed by file stem.

    The authoritative alternative to ranking file mtimes: a copy resets an
    mtime, and a randomised queue makes filename order meaningless, but
    ``AcquisitionDateTime`` is stamped by the instrument. The key is the
    filename without extension, which is what search engines use as the run
    name, so the result joins straight onto a ``sample_info`` frame.
    """
    paths = _resolve(paths)
    out = {}
    for path in paths:
        try:
            meta = read_acquisition_metadata(path)
        except Exception:                                       # noqa: BLE001
            continue
        out[os.path.splitext(os.path.basename(path))[0]] = meta['acquired']
        if verbose:
            print(f'  {os.path.basename(path)}: {out[list(out)[-1]]}')
    return pd.Series(out, name='acquired').sort_values()


def _resolve(paths):
    """Accept a directory, a glob, or a list of paths; return a sorted list."""
    if isinstance(paths, (str, os.PathLike)):
        paths = str(paths)
        if os.path.isdir(paths):
            paths = os.path.join(paths, '*.wiff')
        paths = sorted(_glob.glob(paths))
    return [str(p) for p in paths]
