"""Step 12 — Fiber-type composition of the two muscles, as stacked bars."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import spec_analytics as core
import common as C
C.init()
SEGMENTS = C.TYPE_ORDER + ['mixed']
def readable_on(color):
    """Black or white, whichever reads on `color` (Rec. 601 luminance).
    Computed rather than hand-set so it survives a palette change: type I is a dark
    blue, where black numerals stop reading, while the other four segments are light
    enough to keep them.
    """
    r, g, b = matplotlib.colors.to_rgb(color)
    return 'black' if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else 'white'
FONTSIZE = 10
BAR_WIDTH_IN = 0.38
PANEL_W_IN = 2.1
AXES_H_IN = 2.05
sample_info = C.load_typed()
counts = (sample_info.groupby(['condition1', 'fiber_type']).size()
          .unstack('fiber_type').reindex(index=C.MUSCLE_ORDER, columns=SEGMENTS)
          .fillna(0).astype(int))
totals = counts.sum(axis=1)
percent = 100 * counts.div(totals, axis=0)
print('fibers per muscle and type:')
print(counts.assign(total=totals).to_string())
print('\nas % of the muscle:')
print(percent.round(1).to_string())
for m in C.MUSCLE_ORDER:
    slow = percent.loc[m, 'I'] + percent.loc[m, 'IIa']
    fast = percent.loc[m, 'IIx'] + percent.loc[m, 'IIb']
    print(f'  {m}: slow (I+IIa) {slow:.1f} %, fast (IIx+IIb) {fast:.1f} %, '
          f'mixed {percent.loc[m, "mixed"]:.1f} %  (n = {totals[m]})')
fig, ax = plt.subplots(figsize=(PANEL_W_IN, 2.6))
x = np.arange(len(C.MUSCLE_ORDER))
bottom = np.zeros(len(C.MUSCLE_ORDER))
for seg in SEGMENTS:
    vals = percent[seg].to_numpy()
    ax.bar(x, vals, 0.6, bottom=bottom, color=C.TYPE_COLOR[seg],
           edgecolor='black', linewidth=0.5, zorder=2, label=seg)
    for xi, (v, b, n) in enumerate(zip(vals, bottom, counts[seg].to_numpy())):
        if v >= 7:
            ax.text(xi, b + v / 2, str(n), ha='center', va='center',
                    fontsize=FONTSIZE, zorder=4,
                    color=readable_on(C.TYPE_COLOR[seg]))
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels([f'{m}\nn={totals[m]}' for m in C.MUSCLE_ORDER],
                   fontsize=FONTSIZE)
ax.set_xlim(-0.6, len(C.MUSCLE_ORDER) - 0.4)
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel('Fibers [%]', fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)
C.despine(ax)
fig.tight_layout()
fig.canvas.draw()
ax_w = ax.get_position().width * fig.get_size_inches()[0]
w_data = BAR_WIDTH_IN * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax_w
for patch in ax.patches:
    centre = patch.get_x() + patch.get_width() / 2
    patch.set_width(w_data)
    patch.set_x(centre - w_data / 2)
core.set_axes_size_inches(fig, ax, h_in=AXES_H_IN)
C.save_panel(fig, 'panel_c_fiber_types_by_muscle', {
    'stacked segment': (percent.stack().rename('percent_of_muscle').reset_index()
                        .rename(columns={'condition1': 'muscle',
                                         'level_1': 'fiber_type'})
                        .merge(counts.stack().rename('n_fibers').reset_index()
                               .rename(columns={'condition1': 'muscle',
                                                'level_1': 'fiber_type'}),
                               on=['muscle', 'fiber_type'])),
    'muscle total': totals.rename('n_fibers').reset_index().rename(
        columns={'condition1': 'muscle'}),
})
