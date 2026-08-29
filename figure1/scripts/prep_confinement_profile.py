"""Axial confinement of the SPEC bed, measured per image column."""
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
AI = HERE / '..' / 'data' / 'ai_images'
CHANNEL_IMG = {
    'protein':  AI / 'obj00081_663x1094_8bpc.npy',
    'protease': AI / 'obj00075_624x1097_8bpc.npy',
}
BED_UM = 1000.0
BLUE_INSIDE = 128
ERODE = 4
MIN_DEPTH = 400
MIN_PEAK = 15
WIN = 15
TRACE = 500
PAD = 250
LATERAL_WINDOW_FRAC = (0.02, 0.855)
PAIR_UM = 5.0
def cmyk_to_rgb(a):
    c, m, y, k = [a[..., i].astype(float) / 255 for i in range(4)]
    return np.dstack([255 * (1 - c) * (1 - k), 255 * (1 - m) * (1 - k),
                      255 * (1 - y) * (1 - k)])
def smooth(v, w=WIN):
    return np.convolve(v, np.ones(w) / w, 'same')
def load_channel(name, path):
    """Geometry and signal for one channel, from its own image.
    Returns the signal plane, a per-column (top, bottom) boundary from the blue
    crossing, the pixel size from that image's own bed diameter, and the plug's
    left edge so lateral positions can be expressed in micrometres from it.
    """
    a = np.load(path)
    rgb = cmyk_to_rgb(a)
    ink = a.sum(2) > 0
    upp = BED_UM / ink.sum(1).max()
    inside = rgb[..., 2] < BLUE_INSIDE
    bounds = {}
    for x in range(inside.shape[1]):
        idx = np.flatnonzero(inside[:, x])
        if idx.size:
            bounds[x] = (int(idx[0]), int(idx[-1]))
    sig = rgb[..., 0] if name == 'protein' else rgb[..., 1:].mean(2)
    print(f'{name:9s} {path.name}  bed {ink.sum(1).max()} px = {BED_UM:.0f} um '
          f'({upp:.3f} um/px), {len(bounds)} columns with a boundary')
    return dict(sig=sig, bounds=bounds, upp=upp, left=min(bounds))
def column_profile(sig, top_raw, bot, x):
    """Baseline-subtracted, smoothed profile of one column, or None."""
    if bot - top_raw + 1 < MIN_DEPTH:
        return None
    col = sig[top_raw + ERODE:bot + 1, x].astype(float)
    if col.size < 250:
        return None
    p = col - np.median(col[-int(0.25 * col.size):])
    ps = smooth(p)
    if ps.max() < MIN_PEAK:
        return None
    return p, ps
geom = {name: load_channel(name, path) for name, path in CHANNEL_IMG.items()}
lo_frac, hi_frac = LATERAL_WINDOW_FRAC
out = {}
for name, g in geom.items():
    sig, bounds, upp, left = g['sig'], g['bounds'], g['upp'], g['left']
    crest, width, lateral = [], [], []
    for x, (top_raw, bot) in sorted(bounds.items()):
        if not (lo_frac <= (x - left) * upp / BED_UM <= hi_frac):
            continue
        got = column_profile(sig, top_raw, bot, x)
        if got is None:
            continue
        p, ps = got
        i = int(np.argmax(ps))
        half = ps[i] / 2
        l = i
        while l > 0 and ps[l] >= half:
            l -= 1
        r = i
        while r < ps.size - 1 and ps[r] >= half:
            r += 1
        crest.append(i * upp)
        width.append((r - l) * upp)
        lateral.append((x - left) * upp)
    crest, width = np.array(crest), np.array(width)
    lateral = np.array(lateral)
    out[name] = dict(crest=crest, width=width, lateral=lateral, n=len(width))
    print(f'\n{name}:  {len(width)} columns in the window '
          f'({lo_frac:.3f}-{hi_frac:.3f} of the bed)')
    print(f'   crest depth below own column top: median {np.median(crest):6.0f} um '
          f'(IQR {np.percentile(crest,25):.0f}-{np.percentile(crest,75):.0f})')
    print(f'   FWHM per column:                  median {np.median(width):6.0f} um '
          f'(IQR {np.percentile(width,25):.0f}-{np.percentile(width,75):.0f})')
UPP_REF = float(np.mean([g['upp'] for g in geom.values()]))
depth = (np.arange(TRACE + 2 * PAD) - PAD) * UPP_REF
FLAT_DEG = 20.0
SMOOTH_K = 25
summary = {}
for name in out:
    d = out[name]
    o = np.argsort(d['lateral'])
    x, crest, w = d['lateral'][o], d['crest'][o], d['width'][o]
    cs = np.convolve(crest, np.ones(SMOOTH_K) / SMOOTH_K, 'same')
    tilt = np.degrees(np.arctan(np.abs(np.gradient(cs, x))))
    keep = slice(SMOOTH_K, -SMOOTH_K)
    x, w, tilt = x[keep], w[keep], tilt[keep]
    flat = tilt < FLAT_DEG
    summary[name] = dict(tilt_med=float(np.median(tilt)), n_flat=int(flat.sum()),
                         flat_med=float(np.median(w[flat])),
                         flat_iqr=np.percentile(w[flat], [25, 75]))
    d['flat_lateral'] = x[flat]
    print(f'\n{name}: local band tilt median {np.median(tilt):.0f} deg '
          f'(90th pct {np.percentile(tilt, 90):.0f})')
    print(f'   {flat.sum()} of {len(w)} columns are near-flat '
          f'(< {FLAT_DEG:.0f} deg)')
    print(f'   FWHM there: median {np.median(w[flat]):.0f} um '
          f'(IQR {np.percentile(w[flat], 25):.0f}-{np.percentile(w[flat], 75):.0f})')
def resample(trace_px, upp):
    """Put a per-pixel trace onto the shared micrometre depth axis."""
    src = (np.arange(trace_px.size) - PAD) * upp
    return np.interp(depth, src, trace_px, left=np.nan, right=np.nan)
for name, g in geom.items():
    sig, bounds, upp, left = g['sig'], g['bounds'], g['upp'], g['left']
    flat_lat = set(np.round(out[name]['flat_lateral']).astype(int))
    crest_traces, edge_traces, crest_depths, crest_by_key = [], [], [], {}
    for x, (top_raw, bot) in sorted(bounds.items()):
        lat = (x - left) * upp
        if not (lo_frac <= lat / BED_UM <= hi_frac):
            continue
        got = column_profile(sig, top_raw, bot, x)
        if got is None:
            continue
        p, ps = got
        i = int(np.argmax(ps))
        src = p / ps.max()
        if int(round(lat)) in flat_lat:
            t = np.full(TRACE + 2 * PAD, np.nan)
            s0, s1 = max(0, i - PAD), min(p.size, i + TRACE + PAD)
            t[PAD - (i - s0):PAD + (s1 - i)] = src[s0:s1]
            crest_traces.append(resample(t, upp))
        e = np.full(TRACE + 2 * PAD, np.nan)
        n = min(src.size, TRACE + PAD)
        e[PAD:PAD + n] = src[:n]
        edge_traces.append(resample(e, upp))
        crest_depths.append(i * upp)
        crest_by_key[round(lat / PAIR_UM) * PAIR_UM] = i * upp
    C, E = np.vstack(crest_traces), np.vstack(edge_traces)
    out[name].update(crest_by_key=crest_by_key,
                     med=np.nanmedian(C, axis=0),
                     q1=np.nanpercentile(C, 25, axis=0),
                     q3=np.nanpercentile(C, 75, axis=0),
                     flat_med_trace=np.nanmedian(C, axis=0),
                     flat_q1=np.nanpercentile(C, 25, axis=0),
                     flat_q3=np.nanpercentile(C, 75, axis=0),
                     edge_med_trace=np.nanmedian(E, axis=0),
                     edge_q1=np.nanpercentile(E, 25, axis=0),
                     edge_q3=np.nanpercentile(E, 75, axis=0),
                     crest_depth_med=float(np.median(crest_depths)),
                     crest_depth_iqr=np.percentile(crest_depths, [25, 75]),
                     n_crest=len(crest_traces), n_edge=len(edge_traces))
    print(f'{name:9s} crest-aligned trace from {len(crest_traces)} near-flat '
          f'columns; median crest depth {np.median(crest_depths):.0f} um')
_pa, _pt = out['protein']['crest_by_key'], out['protease']['crest_by_key']
_both = sorted(set(_pa) & set(_pt))
OFFSET_UM = float(np.median([_pt[k] - _pa[k] for k in _both])) if _both else np.nan
_naive = out['protease']['crest_depth_med'] - out['protein']['crest_depth_med']
print(f'\npaired crest offset over {len(_both)} matched positions: '
      f'{OFFSET_UM:+.0f} um (difference of the two medians would read {_naive:+.0f})')
_close = np.mean([abs(_pt[k] - _pa[k]) < 100 for k in _both]) * 100
print(f'   the two crests are within 100 um at {_close:.0f} % of those positions')
out['protein']['draw_depth_um'] = out['protein']['crest_depth_med']
out['protease']['draw_depth_um'] = out['protein']['crest_depth_med'] + OFFSET_UM
print(f"\ndrawn at: protein {out['protein']['draw_depth_um']:.0f} um, "
      f"protease {out['protease']['draw_depth_um']:.0f} um")
for name in out:
    s = summary[name]
    vol = np.pi * (BED_UM / 2) ** 2 * s['flat_med'] / 1e9 * 1e3
    print(f"{name:9s} FWHM {s['flat_med']:.0f} um -> {vol:.0f} nL")
np.savez(HERE / '..' / 'input' / 'confinement_profile.npz', depth=depth,
         paired_offset_um=OFFSET_UM, crests_within_100um_pct=_close,
         **{f'{n}_{k}': out[n][k] for n in out
            for k in ('med', 'q1', 'q3', 'crest', 'width', 'lateral',
                      'flat_med_trace', 'flat_q1', 'flat_q3',
                      'edge_med_trace', 'edge_q1', 'edge_q3',
                      'crest_depth_med', 'crest_depth_iqr', 'n_edge',
                      'n_crest', 'draw_depth_um')},
         **{f'{n}_{k}': summary[n][k] for n in summary
            for k in ('tilt_med', 'n_flat', 'flat_med', 'flat_iqr')})
print('wrote input/confinement_profile.npz')
