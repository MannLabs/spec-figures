"""Annotated MS2 spectrum with a fragment-coverage ladder.

A matplotlib peptide-spectrum panel, so the figure lands in the house style and
the PDF keeps editable text (``pdf.fonttype = 42``). The alternative in this
space is alpharaw's plotly ``PSM_Plot``, which is excellent interactively but
exports raster-ish PDFs whose labels cannot be nudged in Illustrator — and on a
40-residue peptide there are always labels to nudge.

Reading one of these: the ladder above the spectrum is the point. Sticks tell
you what matched; the ladder tells you *where*, and a modification is only
localised when the ladder brackets it on both sides. A phosphosite with c/z
coverage either side of the residue is localised; the same site with only
distant matches is not, however tall the sticks are.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._style import PALETTE_SINGLE, _resolve_panel_size

__all__ = ['plot_annotated_spectrum']

#: Ion series -> colour. N-terminal series (a/b/c) share a blue family and
#: C-terminal (x/y/z) a red one, the convention most spectrum viewers use, so
#: the direction of a series is readable without consulting the legend.
ION_COLORS = {
    'a': '#5B8FF9', 'b': '#3A5FCD', 'c': '#00A0B0',
    'x': '#E8825A', 'y': '#D6403A', 'z': '#E0509A',
}
UNMATCHED_COLOR = '#BBBBBB'


def plot_annotated_spectrum(
    mz, intensity, fragments, *, sequence=None, mod_sites=None,
    tol=20.0, tol_unit='ppm', title=None, figsize=(8, 4),
    label_top_n=40, label_fontsize=6, ladder_fontsize=6.5,
    sequence_fontsize=9, mz_range=None, show_unmatched=True,
    highlight_sites=(), ax=None,
):
    """Draw a centroided MS2 spectrum with matched fragments annotated.

    :param mz, intensity: the spectrum's peaks.
    :param fragments: DataFrame with ``product_mz``, ``ion_type``,
        ``ion_number`` and optionally ``fragment_charge`` — a
        :func:`~..raw.targeted.target_fragments` table for one precursor.
    :param str sequence: peptide sequence; draws the coverage ladder.
    :param highlight_sites: 1-based residue positions to draw in red in the
        ladder (a phosphosite, say).
    :param int label_top_n: annotate only the most intense matched peaks;
        a long peptide matches more fragments than a panel can label.
    :returns: ``(fig, ax, matched)`` where `matched` is one row per matched
        fragment with its observed m/z, intensity and mass error.
    """
    import matplotlib.pyplot as plt

    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if mz_range is not None:
        keep = (mz >= mz_range[0]) & (mz <= mz_range[1])
        mz, intensity = mz[keep], intensity[keep]
    if intensity.size == 0:
        raise ValueError('spectrum is empty in the requested m/z range')

    # --- match theoretical fragments to observed peaks ---------------------
    rows = []
    for _, frag in fragments.iterrows():
        target = float(frag['product_mz'])
        half = (target * tol * 1e-6 if tol_unit == 'ppm' else tol)
        hit = np.where(np.abs(mz - target) <= half)[0]
        if hit.size == 0:
            continue
        best = hit[np.argmax(intensity[hit])]
        rows.append({
            'ion_type': frag['ion_type'],
            'ion_number': int(frag['ion_number']),
            'fragment_charge': int(frag.get('fragment_charge', 1)),
            'theoretical_mz': target,
            'observed_mz': float(mz[best]),
            'intensity': float(intensity[best]),
            'error_ppm': float((mz[best] - target) / target * 1e6),
        })
    matched = pd.DataFrame(rows)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=_resolve_panel_size(figsize))
    else:
        fig = ax.figure

    top = float(intensity.max())
    if show_unmatched:
        ax.vlines(mz, 0, intensity, color=UNMATCHED_COLOR, lw=0.6, zorder=1)
    if len(matched):
        for ion, grp in matched.groupby('ion_type'):
            ax.vlines(grp['observed_mz'], 0, grp['intensity'],
                      color=ION_COLORS.get(ion, '#333333'), lw=1.1, zorder=3,
                      label=ion)
        for _, r in matched.nlargest(min(label_top_n, len(matched)),
                                     'intensity').iterrows():
            charge = '+' * r['fragment_charge']
            ax.annotate(f'{r["ion_type"]}{r["ion_number"]}{charge}',
                        xy=(r['observed_mz'], r['intensity']),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', rotation=90,
                        fontsize=label_fontsize,
                        color=ION_COLORS.get(r['ion_type'], '#333333'))

    ax.set_xlabel('m/z', fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)
    ax.set_ylim(0, top * 1.35)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v / 1000:.0f}k' if v >= 1000
                          else f'{v:.0f}'))
    if title:
        # The ladder is drawn in axes coordinates above y=1, so a default-padded
        # title lands on top of it.
        ax.set_title(title, fontsize=11, fontweight='bold',
                     pad=52 if sequence else 6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    if sequence:
        _draw_ladder(ax, sequence, matched, highlight_sites,
                     sequence_fontsize, ladder_fontsize)
    if created:
        fig.tight_layout()
    return fig, ax, matched


def _draw_ladder(ax, sequence, matched, highlight_sites, seq_fontsize,
                 tick_fontsize):
    """Sequence across the top with a tick at every cleaved bond.

    Drawn in axes coordinates so it sits above the data whatever the y-scale.
    N-terminal ions cut after residue `i`, C-terminal ions before residue
    `n - i + 1`; both are marked on the bond, which is what "localised" means.
    """
    n = len(sequence)
    y_seq = 1.16
    highlight = set(highlight_sites)
    for i, aa in enumerate(sequence):
        x = (i + 0.5) / n
        ax.text(x, y_seq, aa, transform=ax.transAxes, ha='center', va='center',
                fontsize=seq_fontsize, family='monospace',
                color='#D6403A' if (i + 1) in highlight else '#222222',
                fontweight='bold' if (i + 1) in highlight else 'normal',
                clip_on=False)

    if not len(matched):
        return
    for _, r in matched.iterrows():
        ion, num = r['ion_type'], int(r['ion_number'])
        if ion in 'abc':
            bond, y0, y1, va = num, y_seq + 0.05, y_seq + 0.10, 'bottom'
        else:
            bond, y0, y1, va = n - num, y_seq - 0.10, y_seq - 0.05, 'top'
        if not 0 < bond < n:
            continue
        x = bond / n
        colour = ION_COLORS.get(ion, '#333333')
        ax.plot([x, x], [y0, y1], transform=ax.transAxes, color=colour,
                lw=0.9, clip_on=False)
        # Short horizontal flag pointing back along the series direction, so
        # b/y ladders read as brackets rather than a picket fence.
        ax.plot([x, x - 0.4 / n if ion in 'abc' else x + 0.4 / n], [y1, y1]
                if ion in 'abc' else [y0, y0], transform=ax.transAxes,
                color=colour, lw=0.9, clip_on=False)
