"""Step 5 — PCA at the protein-group level, one panel per colouring."""
import pandas as pd

import spec_analytics as core
import common as C

C.init()

df = C.load_filtered()
sample_info = C.load_typed()


def save_pca(stem, color_by, palette, title):
    fig, ax, res = core.plot_pca(
        df, sample_info,
        level='protein',
        color_by=color_by,
        palette=palette,
        figsize=(4, 4),
        point_size=18,
        point_alpha=0.85,
        show_labels=False,
        show_ellipses=True,
        ellipse_confidence=0.95,
        title=title,
    )
    C.unbold(ax)
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.0),
              ncol=len(handles), frameon=False, fontsize=10, handlelength=1.0,
              handletextpad=0.35, borderpad=0.0, columnspacing=1.1)

    # res['transformed'] already carries the PC coordinates merged with
    # sample_info; keep only what the panel actually draws or labels by.
    scores = res['transformed'][['run', 'PC1', 'PC2', 'condition1', 'well',
                                 'fiber_type', 'dominant_myhc']].copy()
    scores['color_by'] = color_by
    var = res['variance_explained']

    C.save_panel(fig, stem, {
        'pca_scores': scores,
        'variance_explained': pd.DataFrame({
            'PC': [f'PC{i}' for i in range(1, len(var) + 1)],
            'fraction_variance_explained': var}),
    })


save_pca('supporting_pca_by_muscle',
         color_by='condition1',
         palette={m: C.MUSCLE_COLOR[m] for m in C.MUSCLE_ORDER},
         title=f'PCA · by muscle · n={sample_info.shape[0]}')

save_pca('panel_e_pca_by_fiber_type',
         color_by='dominant_myhc',
         palette={t: C.TYPE_COLOR[t] for t in C.TYPE_ORDER},
         title=f'PCA · by dominant MyHC · n={sample_info.shape[0]}')
