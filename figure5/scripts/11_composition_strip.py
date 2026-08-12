"""Step 11 — Stand-alone composition strip: the three top rows of the step-03"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import common as C

C.init()

sample_info = C.load_typed()
fiber_order, fractions = C.fiber_sort_order(sample_info)

fig = plt.figure(figsize=(2.7, 3))
gs = GridSpec(3, 1, height_ratios=[0.75, 0.40, 1.00], hspace=0.05,
              left=0.09, right=0.98, top=0.95, bottom=0.04)

ax_typespan = fig.add_subplot(gs[0])
ax_typespan.set_xlim(-0.5, len(fiber_order) - 0.5)
ax_typespan.set_ylim(0, 1)
ax_typespan.axis('off')

ax_muscle = fig.add_subplot(gs[1], sharex=ax_typespan)
muscle = sample_info.set_index('run').loc[fiber_order, 'condition1']
ax_muscle.bar(range(len(fiber_order)), [1] * len(fiber_order),
              color=[C.MUSCLE_COLOR[m] for m in muscle], width=1.0,
              edgecolor='none')
ax_muscle.set_xlim(-0.5, len(fiber_order) - 0.5)
ax_muscle.set_ylim(0, 1)
ax_muscle.set_yticks([])
ax_muscle.set_xticks([])
ax_muscle.set_ylabel('Muscle', rotation=0, ha='right', va='center', fontsize=9)
for s in ('top', 'right', 'left', 'bottom'):
    ax_muscle.spines[s].set_visible(False)
ax_muscle.legend([plt.Rectangle((0, 0), 1, 1, color=C.MUSCLE_COLOR[m])
                  for m in C.MUSCLE_ORDER], C.MUSCLE_ORDER,
                 loc='lower left', bbox_to_anchor=(1.005, 0), frameon=False,
                 fontsize=8, title='Muscle', title_fontsize=8,
                 labelspacing=0.2, handletextpad=0.3, borderaxespad=0)

ax_stack = fig.add_subplot(gs[2], sharex=ax_typespan)
fracs_ordered = fractions.loc[fiber_order].fillna(0)
x = np.arange(len(fiber_order))
bottom = np.zeros(len(fiber_order))
for t in C.TYPE_ORDER:
    ax_stack.bar(x, fracs_ordered[t], bottom=bottom, color=C.TYPE_COLOR[t],
                 label=f'{C.MYH_GENE[t]} ({t})', width=1.0, edgecolor='none')
    bottom += fracs_ordered[t].values
ax_stack.set_ylim(0, 1)
ax_stack.set_xlim(-0.5, len(fiber_order) - 0.5)
ax_stack.set_ylabel('MyHC frac.', fontsize=9)
ax_stack.set_xticks([])
C.despine(ax_stack)
ax_stack.legend(loc='lower left', bbox_to_anchor=(1.005, 0), frameon=False,
                fontsize=8, title='MyHC', title_fontsize=8,
                labelspacing=0.2, handletextpad=0.4, borderaxespad=0)

typed = sample_info.set_index('run').loc[fiber_order, 'fiber_type']
for i0, i1, t in C.type_spans(typed.tolist(), min_n=5):
    ax_typespan.text((i0 + i1 - 1) / 2.0, 0.05, f'{t}\n{i1 - i0}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     linespacing=1.15,
                     color=C.TYPE_COLOR.get(t, 'black'))

C.unbold(ax_stack, ax_muscle)

fiber_index = {r: i for i, r in enumerate(fiber_order)}
sd_stack = (fracs_ordered.rename_axis('run').reset_index()
            .melt(id_vars='run', var_name='fiber_type', value_name='myhc_fraction'))
sd_stack['fiber_index'] = sd_stack['run'].map(fiber_index)
C.save_panel(fig, 'panel_b_fiber_composition', {
    'myhc_fraction': sd_stack,
    'muscle': pd.DataFrame({'run': fiber_order,
                            'fiber_index': range(len(fiber_order)),
                            'muscle': muscle.values,
                            'fiber_type': typed.values}),
})
