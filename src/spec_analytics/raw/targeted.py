"""Targeted quantification from raw files — fragment XICs, areas, calibration.

The workflow behind an MRM-HR / PRM assay, on top of :mod:`.extraction`:

1. turn a target list (``sequence, mods, mod_sites, charge``) into fragment m/z
   with :func:`target_fragments`;
2. build one summed-fragment chromatogram per target per run with
   :func:`summed_fragment_xic`;
3. integrate them all on a **shared, anchored** window with
   :func:`quantify_targets`;
4. fit a calibration curve with :func:`calibration_fit`.

Two things are easy to get wrong and are handled here rather than left to the
caller.

**The integration window must not depend on the concentration.** Peak-picking
each level of a dilution series separately makes the window narrow where the
signal is weak, which bends the calibration curve towards the origin and
flatters the detection limit. :func:`quantify_targets` runs two passes: the
first measures apex and width on a reference level per peptide (by default the
strongest), the second integrates *every* level in a window anchored on that.

**Sum only fragments above the precursor m/z.** The region below the precursor
carries co-isolated precursor isotopes, immonium ions and low-mass background,
and in a wide isolation window that background is not the analyte. Restricting
to y-ions above the precursor is the default filter
(:func:`y_ions_above_precursor`); it is what makes a summed-fragment trace
selective enough to integrate.

**Decoy transitions measure the background you are actually integrating.**
Every target is optionally quantified a second time on the same transitions
shifted in m/z by a few Da, through identical code, over the identical
retention window. Nothing is there, so whatever that yields is what this
integration collects from background and interference at this precursor, in
this run, at this retention time — a per-measurement blank that needs no blank
injection. Two uses:

* ``signal_to_decoy`` per measurement says whether an area is analyte or
  background, which is the question at the bottom of a dilution series;
* the concentration whose fitted area equals the typical decoy area is a
  *measured* detection floor (:func:`calibration_fit` with ``decoy_area=``).
  The conventional 3.3 x SD/slope limit is an extrapolation from the lowest
  standard, so it routinely reports a limit below anything that was ever
  measured, and it inherits all the noise of an SD from a handful of
  replicates. Quote the two together — when they disagree, the decoy one is
  the one that was observed.

The idea and the offset rationale are shared with :mod:`.detect`, which uses
the same trick to calibrate a detection threshold for unidentified precursors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .extraction import (extract_xic, integrate_peak, pick_peak,
                         points_in_window)

__all__ = [
    'SIL_MODIFICATIONS',
    'register_sil_modifications',
    'target_fragments',
    'y_ions_above_precursor',
    'top_n_fragments',
    'summed_fragment_xic',
    'quantify_targets',
    'calibration_fit',
]

#: Stable-isotope labels used by peptide reference standards (Promega 6x5 and
#: friends). alphabase ships the common PTMs but not these, so they have to be
#: registered before a labelled target list can be parsed.
SIL_MODIFICATIONS = {
    'Label:13C(4)15N(1)@T': {'composition': 'C(-4)13C(4)N(-1)15N(1)'},
    'Label:13C(3)15N(1)@S': {'composition': 'C(-3)13C(3)N(-1)15N(1)'},
    'Label:13C(3)15N(1)@A': {'composition': 'C(-3)13C(3)N(-1)15N(1)'},
    'Label:13C(5)15N(1)@V': {'composition': 'C(-5)13C(5)N(-1)15N(1)'},
    'Label:13C(6)15N(1)@L': {'composition': 'C(-6)13C(6)N(-1)15N(1)'},
    'Label:13C(6)15N(1)@I': {'composition': 'C(-6)13C(6)N(-1)15N(1)'},
    'Label:13C(9)15N(1)@F': {'composition': 'C(-9)13C(9)N(-1)15N(1)'},
    'Label:13C(6)15N(2)@K': {'composition': 'C(-6)13C(6)N(-2)15N(2)'},
    'Label:13C(6)15N(4)@R': {'composition': 'C(-6)13C(6)N(-4)15N(4)'},
}

#: Fragment types by activation. CID/HCD cleaves the amide bond to b/y ions;
#: EAD (like ETD/ECD) transfers an electron and gives c/z ions, which is why it
#: keeps labile modifications on the backbone fragments.
FRAGMENT_TYPES = {
    'cid': ['b_z1', 'b_z2', 'y_z1', 'y_z2'],
    'ead': ['c_z1', 'c_z2', 'c_z3', 'z_z1', 'z_z2', 'z_z3'],
}


def register_sil_modifications(extra=None, *, verbose=False):
    """Register the stable-isotope labels in :data:`SIL_MODIFICATIONS`.

    Idempotent, and safe to call at the top of any script that reads a
    labelled target list. ``extra`` adds or overrides entries.
    """
    from alphabase.constants.modification import add_new_modifications

    mods = dict(SIL_MODIFICATIONS)
    if extra:
        mods.update(extra)
    add_new_modifications(mods)
    if verbose:
        print(f'  registered {len(mods)} stable-isotope modifications')
    return mods


def target_fragments(targets, *, activation='cid', frag_types=None,
                     min_mz=100.0, max_mz=2000.0):
    """Fragment m/z for each target peptide, as one long table.

    :param targets: DataFrame with ``sequence``, ``mods``, ``mod_sites`` and
        ``charge`` (alphabase convention: ``mods`` semicolon-separated,
        ``mod_sites`` 1-based, ``0`` for N-term).
    :param str activation: ``'cid'`` (b/y) or ``'ead'`` (c/z); ignored when
        `frag_types` is given explicitly.
    :returns: DataFrame with one row per (target, fragment): ``target_idx``,
        ``sequence``, ``charge``, ``precursor_mz``, ``ion_type``,
        ``ion_number``, ``fragment_charge``, ``product_mz``.

    Call :func:`register_sil_modifications` first if the list carries
    stable-isotope labels, or alphabase will not know their masses.
    """
    from alphabase.peptide.fragment import (create_fragment_mz_dataframe,
                                            get_charged_frag_types)
    from alphabase.peptide.precursor import calc_precursor_mz

    if frag_types is None:
        if activation not in FRAGMENT_TYPES:
            raise ValueError(f'activation must be one of '
                             f'{sorted(FRAGMENT_TYPES)}, got {activation!r}')
        frag_types = FRAGMENT_TYPES[activation]

    prec = targets.copy().reset_index(drop=True)
    prec['target_idx'] = np.arange(len(prec))
    for col in ('mods', 'mod_sites'):
        if col not in prec.columns:
            prec[col] = ''
        prec[col] = prec[col].fillna('').astype(str)
    prec['nAA'] = prec['sequence'].str.len()
    calc_precursor_mz(prec)

    # get_charged_frag_types normalises e.g. ['b_z1','y_z1'] and validates them.
    types = get_charged_frag_types(
        sorted({t.split('_')[0] for t in frag_types}),
        max(int(t.split('_z')[1]) for t in frag_types))
    types = [t for t in types if t in set(frag_types)]
    frag_mz = create_fragment_mz_dataframe(prec, types)

    rows = []
    for _, row in prec.iterrows():
        start, stop = int(row['frag_start_idx']), int(row['frag_stop_idx'])
        block = frag_mz.iloc[start:stop]
        n_aa = int(row['nAA'])
        for col in block.columns:
            ion, charge_tag = col.split('_z')
            fragment_charge = int(charge_tag)
            values = block[col].to_numpy()
            for i, mz in enumerate(values):
                if not np.isfinite(mz) or mz <= 0:
                    continue
                if not (min_mz <= mz <= max_mz):
                    continue
                # alphabase orders N-terminal ions (a/b/c) 1..n-1 along the
                # sequence and C-terminal ions (x/y/z) the same way, so a
                # C-terminal ion's number counts back from the C-terminus.
                number = i + 1 if ion in 'abc' else n_aa - 1 - i
                rows.append({
                    'target_idx': int(row['target_idx']),
                    'sequence': row['sequence'],
                    'charge': int(row['charge']),
                    'precursor_mz': float(row['precursor_mz']),
                    'ion_type': ion,
                    'ion_number': int(number),
                    'fragment_charge': fragment_charge,
                    'product_mz': float(mz),
                })
    return pd.DataFrame(rows)


def y_ions_above_precursor(fragments, *, ion_types=('y', 'z'), margin=0.0):
    """Keep C-terminal fragments heavier than the precursor.

    The default selectivity filter — see the module docstring. `margin` in Th
    pushes the cut further above the precursor.
    """
    keep = (fragments['ion_type'].isin(ion_types)
            & (fragments['product_mz'] > fragments['precursor_mz'] + margin))
    return fragments[keep]


def top_n_fragments(fragments, intensities, n=6):
    """Keep the `n` most intense fragments of each target.

    `intensities` is a same-length array (e.g. apex heights from a reference
    run). Use when a library intensity ranking is available and you would
    rather sum a few strong transitions than everything.
    """
    work = fragments.copy()
    work['_intensity'] = np.asarray(intensities, dtype=float)
    out = (work.sort_values('_intensity', ascending=False)
           .groupby('target_idx', sort=False).head(n)
           .drop(columns='_intensity'))
    return out.sort_index()


def summed_fragment_xic(ms_data, fragments, *, rt_range=None, mz_tol=20.0,
                        mz_tol_unit='ppm'):
    """One chromatogram per target: its fragments' XICs summed.

    :param fragments: rows of a :func:`target_fragments` table for ONE target
        (already filtered — see :func:`y_ions_above_precursor`).
    :returns: ``(rt, intensity)`` arrays, empty when nothing was extracted.

    Summing gives the trace its selectivity: an interference has to coelute
    with, and be present in, several transitions at once to survive.
    """
    if len(fragments) == 0:
        return np.array([]), np.array([])
    precursor_mz = float(fragments['precursor_mz'].iloc[0])
    xic = extract_xic(
        ms_data, fragments['product_mz'].to_numpy(),
        mz_tol=mz_tol, mz_tol_unit=mz_tol_unit, rt_range=rt_range,
        ms_level=2, precursor_mz=precursor_mz)
    return xic['rt'].to_numpy(), xic['intensity'].to_numpy()


def quantify_targets(ms_files, fragments, *, rt_range=None, mz_tol=20.0,
                     mz_tol_unit='ppm', fragment_filter=y_ions_above_precursor,
                     reference_by='sequence', reference_selector='max_signal',
                     integrate_fwhm=1.5, search_fwhm=1.5, points_fwhm=1.27,
                     max_rt_delta_fwhm=1.0, min_sn=3.0, interpolate_points=500,
                     baseline='min_boundary', decoy_offset_da=6.7,
                     verbose=True):
    """Integrate every target in every run, on an anchored window.

    :param dict ms_files: ``{run_name: ms_data}`` (see
        :func:`~.extraction.load_ms_data`).
    :param fragments: a :func:`target_fragments` table for all targets.
    :param str reference_by: column grouping targets that share a peak — for a
        dilution series of one peptide at several label states, ``'sequence'``.
        Pass None to anchor each target on itself.
    :param reference_selector: ``'max_signal'`` (default) anchors on the target
        of the group with the strongest apex in that run, or an int
        ``target_idx``.
    :param float integrate_fwhm: half-window for integration, in FWHM units.
    :param float search_fwhm: half-window around the anchor apex within which
        each target's own apex is re-found, in FWHM units. Absorbs small
        retention shifts between label states without letting a weak target
        wander onto an interference.
    :param float points_fwhm: half-window for ``points_across_peak``, in FWHM
        units. The default 1.27 is the conventional **6 sigma** span, since
        FWHM = 2.355 sigma and 1.27 x 2.355 ~ 3 sigma each side. Quote which
        span a points-per-peak number used — halving it halves the count, and
        the usual "at least 6-10 points across the peak" rule of thumb assumes
        the full peak, not the FWHM.
    :param float max_rt_delta_fwhm: a target whose own apex lands further than
        this many FWHM from the group anchor is flagged ``rt_outlier``. It is
        **flagged, not dropped** — deciding to exclude a measurement is the
        analyst's call and belongs in the record.

        This is the honest replacement for the hand-written exclusion list that
        targeted workflows accumulate. A label state at the bottom of a
        dilution series has little signal of its own, so if a co-eluting
        interference sits nearby the window latches onto it: the tell is an
        apex displaced from where every other level of the same peptide sits,
        together with an area that breaks the dilution ladder. Seen on a 6x5
        standard, where the 20 amol level of one peptide read five times the
        200 amol level with its apex 3.2 s (1.2 FWHM) off.

    :param decoy_offset_da: m/z shift for the decoy transitions, or None to
        skip them. Must be larger than the extraction tolerance, and should not
        be a plausible neutral loss or an isotope spacing — 6.7 Da is none of
        those. The decoys are integrated over the same window as the target, so
        ``decoy_area`` is the background this measurement collects and
        ``signal_to_decoy`` is how far above it the analyte sits.

    :returns: one row per (run, target) with apex, fwhm, area, S/N, points
        across the peak, ``rt_outlier``, ``decoy_area``, ``signal_to_decoy``,
        and which reference anchored it.

    Why the anchor: see the module docstring. The reference peak's FWHM is also
    what sets the window width, so all levels of a series are integrated over
    the same span of retention time.
    """
    frag_by_target = dict(tuple(fragments.groupby('target_idx', sort=True)))
    meta = (fragments.drop_duplicates('target_idx')
            .set_index('target_idx')[['sequence', 'charge', 'precursor_mz']])
    groups = (meta.groupby(reference_by).groups if reference_by
              else {t: [t] for t in meta.index})

    rows = []
    for run, ms_data in ms_files.items():
        # --- pass 1: the anchor peak per group -----------------------------
        traces, decoy_traces = {}, {}
        for target_idx, frags in frag_by_target.items():
            if fragment_filter is not None:
                frags = fragment_filter(frags)
            traces[target_idx] = summed_fragment_xic(
                ms_data, frags, rt_range=rt_range, mz_tol=mz_tol,
                mz_tol_unit=mz_tol_unit)
            if decoy_offset_da:
                # Identical code, identical scans, identical tolerance — only
                # the product m/z move. The precursor is left alone so the
                # decoys are drawn from the very same spectra.
                decoy = frags.copy()
                decoy['product_mz'] = decoy['product_mz'] + decoy_offset_da
                decoy_traces[target_idx] = summed_fragment_xic(
                    ms_data, decoy, rt_range=rt_range, mz_tol=mz_tol,
                    mz_tol_unit=mz_tol_unit)

        anchors = {}
        for key, members in groups.items():
            members = list(members)
            if reference_selector == 'max_signal':
                ref = max(members, key=lambda t: (traces[t][1].max()
                                                  if traces[t][1].size else -1))
            else:
                ref = int(reference_selector)
            rt, inten = traces[ref]
            peak = pick_peak(rt, inten, baseline='p5', min_sn=min_sn,
                             interpolate_points=interpolate_points)
            anchors[key] = (ref, peak)

        # --- pass 2: integrate every target on its group's anchor ----------
        for key, members in groups.items():
            ref, anchor = anchors[key]
            for target_idx in members:
                rt, inten = traces[target_idx]
                info = meta.loc[target_idx]
                row = {'run': run, 'target_idx': int(target_idx),
                       'sequence': info['sequence'], 'charge': int(info['charge']),
                       'precursor_mz': float(info['precursor_mz']),
                       'reference_target_idx': int(ref),
                       'anchor_rt': anchor['apex_rt'], 'fwhm': anchor['fwhm']}
                if rt.size == 0 or not np.isfinite(anchor['apex_rt']):
                    rows.append({**row, 'apex_rt': np.nan, 'rt_delta_s': np.nan,
                                 'area': 0.0, 'sn': np.nan, 'n_points': 0,
                                 'points_across_peak': 0, 'rt_outlier': False,
                                 'decoy_area': np.nan,
                                 'signal_to_decoy': np.nan})
                    continue

                half = search_fwhm * anchor['fwhm']
                local = pick_peak(rt, inten,
                                  rt_range=(anchor['apex_rt'] - half,
                                            anchor['apex_rt'] + half),
                                  baseline='p5',
                                  interpolate_points=interpolate_points)
                # A weak level may have no apex of its own; fall back to the
                # anchor's rather than to the window maximum, which at low
                # signal is noise.
                apex = (local['apex_rt'] if np.isfinite(local['apex_rt'])
                        else anchor['apex_rt'])
                width = integrate_fwhm * anchor['fwhm']
                integ = integrate_peak(rt, inten, apex - width, apex + width,
                                       baseline=baseline)

                decoy_area = np.nan
                if decoy_offset_da:
                    d_rt, d_int = decoy_traces[target_idx]
                    if d_rt.size:
                        decoy_area = integrate_peak(
                            d_rt, d_int, apex - width, apex + width,
                            baseline=baseline)['area']
                    else:
                        # No decoy signal at all is a background of zero, not
                        # an unknown one — the honest value here is 0.
                        decoy_area = 0.0

                rows.append({
                    **row,
                    'apex_rt': float(apex),
                    'rt_delta_s': float((apex - anchor['apex_rt']) * 60.0),
                    'area': integ['area'],
                    'peak_baseline': integ['baseline'],
                    'sn': local['sn'],
                    'n_points': integ['n_points'],
                    'points_across_peak': points_in_window(
                        rt, apex, points_fwhm * anchor['fwhm']),
                    'rt_outlier': bool(
                        anchor['fwhm'] > 0
                        and abs(apex - anchor['apex_rt'])
                        > max_rt_delta_fwhm * anchor['fwhm']),
                    'decoy_area': decoy_area,
                    'signal_to_decoy': (integ['area'] / decoy_area
                                        if decoy_area and decoy_area > 0
                                        else np.inf),
                })
        if verbose:
            print(f'  quantify_targets: {run} done '
                  f'({len(frag_by_target)} targets)', flush=True)
    return pd.DataFrame(rows)


def calibration_fit(concentration, area, *, weighting='1/x2', decoy_area=None,
                    decoy_multiple=3.0):
    """Least-squares calibration line with LOD/LOQ and per-point accuracy.

    :param str weighting: ``'1/x2'`` (default) or ``'none'``. A calibration
        spanning orders of magnitude must be weighted: unweighted least squares
        minimises absolute residuals, so the top standard alone fixes the line
        and the low end — the part that decides the detection limit — is
        fitted to whatever is left over. 1/x² equalises the *relative*
        residuals, which is what a constant-CV measurement actually has.

    LOD and LOQ are 3.3 and 10 times the SD of the lowest level over the slope
    (ICH Q2). They are only as good as that SD: with few replicates it is a
    noisy estimate, and the limits inherit the noise. They are also an
    *extrapolation* — nothing was measured down there — so they can and often
    do land below the lowest standard on the curve.

    :param decoy_area: areas from decoy transitions (see
        :func:`quantify_targets`), for the same peptide. Their median is the
        background this integration collects, and the concentration whose
        fitted area equals ``decoy_multiple`` times it is reported as
        ``lod_decoy``: a detection floor that was **measured** rather than
        extrapolated. Quote it beside ``lod``; where they disagree, this is the
        one backed by an observation.

        ``lod_decoy`` is NaN when the decoy background is zero. That is not a
        failure — on a clean synthetic standard in centroided spectra there is
        often nothing whatsoever within tolerance of the shifted transitions,
        which says the peaks are background-free. It simply means there is no
        measured floor to quote, and the SD-based ``lod`` is what you have.

    :returns: dict with ``slope, intercept, r_squared, weighting, lod, loq,
        lod_decoy, decoy_area_median, n_levels`` and a per-level ``levels``
        frame carrying mean, SD, CV and accuracy (back-calculated
        concentration over nominal).
    """
    x = np.asarray(concentration, dtype=float)
    y = np.asarray(area, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        raise ValueError('need at least two finite (concentration, area) pairs')

    if weighting == '1/x2':
        w = 1.0 / np.square(x)
    elif weighting == 'none':
        w = np.ones_like(x)
    else:
        raise ValueError(f"weighting must be '1/x2' or 'none', got {weighting!r}")

    sw = w.sum()
    sx, sy = (w * x).sum(), (w * y).sum()
    sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
    denom = sw * sxx - sx * sx
    # Compared against the scale of its own terms, not against zero: with all
    # concentrations equal the two products cancel only to rounding, and an
    # exact == 0 test would let a meaningless fit through with a huge slope.
    if abs(denom) <= 1e-12 * max(sw * sxx, sx * sx):
        raise ValueError('calibration is degenerate — all concentrations equal?')
    slope = (sw * sxy - sx * sy) / denom
    intercept = (sxx * sy - sx * sxy) / denom

    pred = slope * x + intercept
    ss_res = (w * np.square(y - pred)).sum()
    ss_tot = (w * np.square(y - (w * y).sum() / sw)).sum()
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    levels = (pd.DataFrame({'concentration': x, 'area': y})
              .groupby('concentration')['area']
              .agg(['mean', 'std', 'count']).reset_index())
    levels['cv_pct'] = levels['std'] / levels['mean'] * 100.0
    # Back-calculated concentration, which is what "accuracy" means for a
    # calibration point — not the ratio of areas.
    levels['measured_concentration'] = (levels['mean'] - intercept) / slope
    levels['accuracy_pct'] = (levels['measured_concentration']
                              / levels['concentration'] * 100.0)

    lowest = levels.loc[levels['concentration'].idxmin()]
    sd_low = float(lowest['std']) if np.isfinite(lowest['std']) else np.nan
    lod = 3.3 * sd_low / slope if slope > 0 else np.nan
    loq = 10.0 * sd_low / slope if slope > 0 else np.nan

    decoy_median, lod_decoy = np.nan, np.nan
    if decoy_area is not None:
        d = np.asarray(decoy_area, dtype=float)
        d = d[np.isfinite(d)]
        if d.size:
            decoy_median = float(np.median(d))
            # A zero decoy background is a real and welcome result — nothing at
            # all within tolerance of the shifted transitions — but there is
            # then no measured floor to report, and solving the line for it
            # would just return -intercept/slope, a number about the intercept
            # rather than about the background. Leave it undefined and let the
            # SD-based limit stand.
            if slope > 0 and decoy_median > 0:
                candidate = (decoy_multiple * decoy_median - intercept) / slope
                lod_decoy = float(candidate) if candidate > 0 else np.nan

    return {'slope': float(slope), 'intercept': float(intercept),
            'r_squared': r_squared, 'weighting': weighting,
            'lod': float(lod), 'loq': float(loq),
            'sd_lowest_level': sd_low,
            'lod_decoy': lod_decoy, 'decoy_area_median': decoy_median,
            'decoy_multiple': float(decoy_multiple),
            'n_levels': int(len(levels)), 'levels': levels}
