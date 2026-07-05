import numpy as np
import matplotlib.pyplot as plt

from .utils import K_COLORS, save_plot

TABU_LABEL = "facteur d'exploration du tabou"


def plot_score_vs_alpha(df):
    for N, n_df in df.groupby('N'):
        Ks = sorted(n_df['K'].unique())

        fig, ax = plt.subplots(figsize=(6, 4))

        for K in Ks:
            k_df = n_df[n_df['K'] == K].groupby('alpha')['score'].agg(['mean', 'std']).reset_index()
            k_df = k_df.sort_values('alpha')
            color = K_COLORS.get(K, 'black')
            ax.errorbar(k_df['alpha'], k_df['mean'], yerr=k_df['std'],
                        marker='o', label=f"K={K}", color=color, capsize=3)

        ax.set_xscale('symlog', linthresh=1e-4)
        ax.set_xlabel('alpha')
        ax.set_ylabel('Score agrégé')
        ax.legend()

        plt.tight_layout()
        save_plot("score_vs_alpha", f"N{N}", subdir=f"rank/{N}", overwrite=True)
        plt.close()


def plot_heatmap_alpha_tabu(df):
    for N, n_df in df.groupby('N'):
        Ks = sorted(n_df['K'].unique())
        alphas = sorted(n_df['alpha'].unique())
        tabu_factors = sorted(n_df['tabu_factor'].unique())

        n_cols = len(Ks)
        cell_size = 0.6
        fig_w = max(4, cell_size * len(alphas) + 1.5) * n_cols
        fig_h = max(3.5, cell_size * len(tabu_factors) + 1.5)
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
        if n_cols == 1:
            axes = [axes]

        show_text = len(tabu_factors) * len(alphas) <= 80

        vmin = n_df['score'].min()
        vmax = n_df['score'].max()

        for ax, K in zip(axes, Ks):
            k_df = n_df[n_df['K'] == K]

            grid = np.full((len(tabu_factors), len(alphas)), np.nan)
            for i, tf in enumerate(tabu_factors):
                for j, a in enumerate(alphas):
                    row = k_df[(k_df['tabu_factor'] == tf) & (k_df['alpha'] == a)]
                    if not row.empty:
                        grid[i, j] = row['score'].iloc[0]

            im = ax.imshow(grid, aspect='auto', vmin=vmin, vmax=vmax)

            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=45, ha='right')
            ax.set_yticks(range(len(tabu_factors)))
            ax.set_yticklabels(tabu_factors)
            ax.set_title(f"K={K}")
            ax.set_xlabel('alpha')
            ax.set_ylabel(TABU_LABEL)

            if show_text:
                for i in range(len(tabu_factors)):
                    for j in range(len(alphas)):
                        if not np.isnan(grid[i, j]):
                            ax.text(j, i, f"{grid[i, j]:.4f}", ha='center', va='center', fontsize=8)

            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        save_plot("heatmap_alpha_tabu", f"N{N}", subdir=f"rank/{N}", overwrite=True)
        plt.close()
