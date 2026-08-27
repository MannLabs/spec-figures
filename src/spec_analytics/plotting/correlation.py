"""Correlation scatter and the QC protein-intensity heatmap."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ._style import (PALETTE, PALETTE_SINGLE, _hide_top_right_spines, _LEVEL_COLS, _resolve_highlights)
def plot_correlation(
    df,
    sample_info,
    *,
    level='protein',
    group_col='condition2',
    condition_a=None,
    condition_b=None,
    conditions=None,
    figsize=None,
    color=None,
    point_size=8,
    alpha=0.6,
    color_by_density=True,
    density_cmap='inferno',
    highlight_genes=None,
    highlight_ids=None,
    highlight_color='#FFD700',
    highlight_size=60,
    label_highlighted=True,
    show_diagonal=True,
    title=None,
):
    """Scatter of log2 mean intensity per pair of conditions.
    Single-pair mode (default when `condition_a`/`condition_b` are given OR
    `group_col` has exactly two unique values): one axes with Pearson r,
    Spearman rho and y=x diagonal. Returns `(fig, ax, plot_df)`.
    Grid mode: when more than two conditions exist in `group_col` and you
    don't pass `condition_a`/`condition_b`, an N x N grid of scatter plots is
    drawn (lower triangle filled, diagonal labelled with the condition name,
    upper triangle empty). Use `conditions=[...]` to restrict / reorder which
    conditions appear. Returns `(fig, axes, summary_df)` where `summary_df`
    has one row per pair with `pearson_r`, `spearman_rho`, `n_shared`.
    Background points are colour-coded by 2D density (`color_by_density=True`,
    default) using a Gaussian KDE.
    Highlight specific entities with `highlight_genes=...` (case-insensitive,
    resolved via the `genes` column) or `highlight_ids=...` (matches the
    entity id directly).
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr, gaussian_kde
    if level not in _LEVEL_COLS:
        raise ValueError(f"level must be 'protein' | 'peptide' | 'precursor', got {level!r}")
    id_col, val_col = _LEVEL_COLS[level]
    available_groups = list(sample_info[group_col].dropna().unique())
    explicit_pair = (condition_a is not None and condition_b is not None)
    if explicit_pair:
        ordered = [condition_a, condition_b]
        single_pair = True
    elif conditions is not None:
        ordered = list(conditions)
        single_pair = (len(ordered) == 2)
    elif len(available_groups) == 2:
        ordered = available_groups
        single_pair = True
    else:
        ordered = available_groups
        single_pair = False
    missing = [c for c in ordered if c not in available_groups]
    if missing:
        raise ValueError(
            f'condition(s) {missing!r} not found in sample_info[{group_col!r}] '
            f'(available: {available_groups!r})'
        )
    sub_df = df.dropna(subset=[id_col, val_col])
    sub_df = sub_df[sub_df[val_col] > 0]
    means_by_condition = {}
    for cond in ordered:
        runs = set(sample_info.loc[sample_info[group_col] == cond, 'run'])
        vals = sub_df[sub_df['run'].isin(runs)]
        means_by_condition[cond] = (
            np.log2(vals[val_col]).groupby(vals[id_col]).mean()
        )
    highlight_set, label_map = _resolve_highlights(
        df, id_col, highlight_genes, highlight_ids
    )
    def _draw_panel(ax, cond_a, cond_b, *, show_axis_labels=True,
                    show_legend_box=True):
        """Draw a single pairwise scatter on `ax`. Returns (n, r, rho, plot_df)."""
        ma = means_by_condition[cond_a]
        mb = means_by_condition[cond_b]
        common = ma.index.intersection(mb.index)
        if len(common) < 2:
            ax.text(0.5, 0.5, f'n = {len(common)}\n(insufficient overlap)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='gray')
            ax.set_xticks([]); ax.set_yticks([])
            return 0, float('nan'), float('nan'), pd.DataFrame()
        a = ma.loc[common].to_numpy()
        b = mb.loc[common].to_numpy()
        pearson_r, _ = pearsonr(a, b)
        spearman_r, _ = spearmanr(a, b)
        col_a = f'log2_mean_{cond_a}'
        col_b = f'log2_mean_{cond_b}'
        panel_df = pd.DataFrame({id_col: common, col_a: a, col_b: b})
        panel_df['highlighted'] = panel_df[id_col].isin(highlight_set)
        panel_df['label'] = panel_df[id_col].map(label_map).fillna('')
        bg = panel_df[~panel_df['highlighted']]
        if color_by_density and len(bg) >= 2:
            bg_a = bg[col_a].to_numpy()
            bg_b = bg[col_b].to_numpy()
            try:
                kde = gaussian_kde(np.vstack([bg_a, bg_b]))
                density = kde(np.vstack([bg_a, bg_b]))
            except np.linalg.LinAlgError:
                density = np.ones_like(bg_a)
            order = density.argsort()
            ax.scatter(bg_a[order], bg_b[order],
                       c=density[order], s=point_size, alpha=alpha,
                       cmap=density_cmap, edgecolor='none')
        else:
            ax.scatter(bg[col_a], bg[col_b],
                       s=point_size, alpha=alpha,
                       c=color or PALETTE_SINGLE[0], edgecolor='none')
        if show_diagonal:
            lo = float(min(a.min(), b.min()))
            hi = float(max(a.max(), b.max()))
            ax.plot([lo, hi], [lo, hi], color='black',
                    linestyle='--', linewidth=1.0, alpha=0.6, zorder=1)
        if panel_df['highlighted'].any():
            hi_df = panel_df[panel_df['highlighted']]
            ax.scatter(hi_df[col_a], hi_df[col_b],
                       s=highlight_size, c=highlight_color,
                       edgecolor='black', linewidth=0.6, alpha=1.0, zorder=10)
            if label_highlighted:
                for _, row in hi_df.iterrows():
                    ax.annotate(row['label'] or str(row[id_col]),
                                (row[col_a], row[col_b]),
                                fontsize=8, fontweight='bold',
                                xytext=(5, 5), textcoords='offset points',
                                zorder=11)
        if show_legend_box:
            txt = (f'r = {pearson_r:.3f}\n'
                   f'{chr(961)} = {spearman_r:.3f}\n'
                   f'n = {len(common):,}')
            ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.9))
        if show_axis_labels:
            ax.set_xlabel(f'log₂ mean {level} ({cond_a})',
                          fontsize=10, fontweight='bold')
            ax.set_ylabel(f'log₂ mean {level} ({cond_b})',
                          fontsize=10, fontweight='bold')
        return len(common), float(pearson_r), float(spearman_r), panel_df
    if single_pair:
        cond_a, cond_b = ordered
        fig, ax = plt.subplots(figsize=figsize or (6, 6))
        n, pr, sp, panel_df = _draw_panel(ax, cond_a, cond_b)
        if title is None:
            title = f'{cond_a} vs {cond_b}'
        ax.set_title(title, fontsize=13, fontweight='bold')
        _hide_top_right_spines(ax)
        plt.tight_layout()
        return fig, ax, panel_df
    N = len(ordered)
    if N < 2:
        raise ValueError(f'need >= 2 conditions for a grid, got {N}')
    cell = 3.0
    fig, axes = plt.subplots(N, N, figsize=figsize or (cell * N, cell * N),
                             squeeze=False)
    summary_rows = []
    for i in range(N):
        for j in range(N):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, str(ordered[i]),
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=12, fontweight='bold')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            if j > i:
                ax.set_visible(False)
                continue
            cond_a = ordered[j]
            cond_b = ordered[i]
            n, pr, sp, _ = _draw_panel(
                ax, cond_a, cond_b,
                show_axis_labels=False, show_legend_box=True,
            )
            summary_rows.append({
                'condition_a': cond_a, 'condition_b': cond_b,
                'n_shared': n, 'pearson_r': pr, 'spearman_rho': sp,
            })
            if i != N - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(str(cond_a), fontsize=10, fontweight='bold')
            if j != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(str(cond_b), fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=8)
    if title is None:
        title = f'Pairwise {level}-level correlation'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes, pd.DataFrame(summary_rows)
def plot_qc_protein_heatmap(
    df,
    sample_info,
    protein_groups,
    *,
    group_col='condition2',
    figsize=None,
    cmap='Reds',
    vmin=None,
    vmax=None,
    cbar_label='log₂ mean intensity',
    missing_color='#dddddd',
    annot=False,
    annot_decimals=1,
    annot_fontsize=9,
    group_label_fontsize=11,
    group_label_palette=None,
    title=None,
):
    """QC heatmap of selected marker proteins across conditions.
    Rows = unique values of `sample_info[group_col]` (default `condition2`).
    Columns = the genes in `protein_groups`, ordered by category and separated
    visually by vertical dividers; the category name is drawn above each block.
    Cell value = log2 mean `pg_intensity` across the runs in that condition,
    averaged over all `protein_group` rows whose first gene name matches.
    Missing measurements are drawn in `missing_color` (default light grey).
    `protein_groups` is a dict[str, list[str]] mapping category name to a list
    of gene symbols. Define it at the call site (typically in a notebook cell)
    so the panel composition is visible right next to the plot.
    Returns `(fig, ax, matrix)` where `matrix` is the (condition x gene)
    DataFrame of log2 values that was plotted.
    """
    import matplotlib.pyplot as plt
    if not protein_groups:
        raise ValueError('protein_groups must be a non-empty dict')
    ordered_genes = []
    boundaries = []
    cat_centers = []
    cat_palette = group_label_palette if group_label_palette is not None else PALETTE
    for ci, (cat, genes) in enumerate(protein_groups.items()):
        start = len(ordered_genes)
        for g in genes:
            ordered_genes.append(g.upper())
        end = len(ordered_genes)
        if start > 0:
            boundaries.append(start)
        cat_centers.append((cat, (start + end - 1) / 2.0,
                            cat_palette[ci % len(cat_palette)]))
    if not ordered_genes:
        raise ValueError('protein_groups is empty')
    conds = list(sample_info[group_col].dropna().unique())
    cond_runs = {
        c: set(sample_info.loc[sample_info[group_col] == c, 'run'])
        for c in conds
    }
    df_pg = df.dropna(subset=['protein_group', 'pg_intensity'])
    df_pg = df_pg.assign(
        _gene=df_pg['genes'].astype(str).str.split(';').str[0].str.upper()
    )
    matrix = pd.DataFrame(np.nan, index=conds, columns=ordered_genes, dtype=float)
    for gene in ordered_genes:
        sub = df_pg[df_pg['_gene'] == gene]
        if sub.empty:
            continue
        for cond in conds:
            cond_sub = sub[sub['run'].isin(cond_runs[cond])]
            if cond_sub.empty:
                continue
            pg_means = (cond_sub.groupby('protein_group')['pg_intensity']
                                .mean())
            pg_means = pg_means[pg_means > 0]
            if pg_means.empty:
                continue
            matrix.loc[cond, gene] = float(np.log2(pg_means.mean()))
    if figsize is None:
        figsize = (max(6, 0.55 * len(ordered_genes) + 1.5), 1.2 + 0.6 * len(conds))
    fig, ax = plt.subplots(figsize=figsize)
    masked = np.ma.masked_invalid(matrix.to_numpy())
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(missing_color)
    if vmin is None:
        vmin = float(np.nanmin(matrix.to_numpy())) if np.isfinite(matrix.to_numpy()).any() else 0
    if vmax is None:
        vmax = float(np.nanmax(matrix.to_numpy())) if np.isfinite(matrix.to_numpy()).any() else 1
    im = ax.imshow(masked, aspect='auto', cmap=cmap_obj, vmin=vmin, vmax=vmax)
    if annot:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix.iat[i, j]
                if np.isfinite(v):
                    rgba = cmap_obj((v - vmin) / max(vmax - vmin, 1e-12))
                    luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = 'white' if luma < 0.55 else 'black'
                    ax.text(j, i, f'{v:.{annot_decimals}f}',
                            ha='center', va='center',
                            fontsize=annot_fontsize, color=text_color)
    ax.set_xticks(range(len(ordered_genes)))
    ax.set_xticklabels(ordered_genes, rotation=45, ha='right')
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.tick_params(axis='both', length=0)
    for b in boundaries:
        ax.axvline(b - 0.5, color='black', linewidth=1.2, zorder=3)
    for cat, centre, color in cat_centers:
        ax.text(centre, -0.6, cat,
                ha='center', va='bottom',
                fontsize=group_label_fontsize, fontweight='bold',
                color=color, transform=ax.transData)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11, fontweight='bold')
    if title is None:
        title = f'QC marker proteins by {group_col}'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=24)
    plt.tight_layout()
    return fig, ax, matrix
