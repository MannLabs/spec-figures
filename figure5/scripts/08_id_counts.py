"""Step 8 — Per-fiber ID counts by fiber type, one panel per metric."""
import matplotlib.pyplot as plt
import pandas as pd

import spec_analytics as core
import common as C

C.init()

df = C.load_filtered(columns=['run', 'protein_group', 'precursor_id',
                              'precursor_intensity'])
sample_info = C.load_typed()

det = df[df['precursor_intensity'].notna()]
counts = det.groupby('run').agg(
    n_protein_groups=('protein_group', 'nunique'),
    n_precursors=('precursor_id', 'nunique'),
).reset_index()
counts = counts.merge(sample_info[['run', 'condition1', 'well', 'fiber_type']],
                      on='run', how='inner')

order = C.TYPE_ORDER + ['mixed']
counts['fiber_type'] = pd.Categorical(counts['fiber_type'], categories=order,
                                      ordered=True)
counts = counts.sort_values('fiber_type')
counts.to_csv(C.DATA / 'per_fiber_counts.csv', index=False)

print('Summary by fiber type:')
print(counts.groupby('fiber_type', observed=True).agg(
    n=('run', 'count'),
    median_pg=('n_protein_groups', 'median'),
    median_pr=('n_precursors', 'median'),
).round(0).to_string())


def render(stem, y_col, y_label, title):
    core.plot_boxplot_with_points(
        counts, x_col='fiber_type', y_col=y_col,
        # The pre-compensated dict: seaborn desaturates it to 0.75, landing on
        # C.TYPE_COLOR, so this panel and the raw-matplotlib panels agree.
        palette=C.TYPE_COLOR_BOX,
        figsize=(4, 4),
        y_label=y_label, x_label='Fiber type', title=title,
        ymin=0, box_width=0.65,
        point_size=6, point_alpha=0.7, point_color='black',
        # Off: the per-type medians belong in the figure legend, not on the boxes.
        # They stay in the printout and in `<stem>_sourcedata.csv`.
        show_median_labels=False,
        xtick_rotation=0,
    )
    fig = plt.gcf()
    C.unbold(*fig.axes)
    C.save_panel(fig, stem, counts[['run', 'condition1', 'well', 'fiber_type',
                                    y_col]].rename(columns={y_col: 'value'})
                 .assign(metric=y_col))


render('panel_d_protein_groups_per_fiber', 'n_protein_groups',
       'Protein groups per fiber', f'Protein groups · n={len(counts)} fibers')
render('supporting_precursors_per_fiber', 'n_precursors',
       'Precursors per fiber', f'Precursors · n={len(counts)} fibers')
