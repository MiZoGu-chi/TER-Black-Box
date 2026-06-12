import numpy as np
import matplotlib.pyplot as plt

from .utils import figure, K_COLORS


def plot_aggregated_fidelity(final_scatter_by_K, subdir=None, tag=None):
    with figure("model_fidelity", tag, subdir=subdir):
        plt.figure(figsize=(8, 8))
        all_preds, all_reals = [], []

        for K, scatter_list in final_scatter_by_K.items():
            scatter = np.vstack(scatter_list)
            preds, reals = scatter[:, 0], scatter[:, 1]
            all_preds.extend(preds)
            all_reals.extend(reals)
            plt.scatter(reals, preds, s=5, color=K_COLORS.get(K, 'black'), alpha=0.3, label=f"K={K}")

        if all_preds:
            min_val = min(min(all_reals), min(all_preds))
            max_val = max(max(all_reals), max(all_preds))
            plt.plot([min_val, max_val], [min_val, max_val], 'b--')

        plt.xlabel("Fitness réelle", fontsize=12)
        plt.ylabel("Fitness prédite", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)


def plot_aggregated_fidelity_with_time(fidelity_samples_by_K, subdir=None, tag=None):
    sorted_K = [k for k in sorted(fidelity_samples_by_K.keys()) if len(fidelity_samples_by_K[k]) > 0]
    if not sorted_K:
        return
    with figure("model_training_fidelity", tag, subdir=subdir):
        fig, axes = plt.subplots(1, len(sorted_K), figsize=(7 * len(sorted_K), 6))
        if len(sorted_K) == 1:
            axes = [axes]

        for ax, K in zip(axes, sorted_K):
            samples = fidelity_samples_by_K[K]
            preds   = [s[0] for s in samples]
            reals   = [s[1] for s in samples]
            n_evals = [s[2] for s in samples]

            sc = ax.scatter(reals, preds, c=n_evals, cmap='viridis', s=8, alpha=0.6)
            min_val, max_val = min(min(reals), min(preds)), max(max(reals), max(preds))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            fig.colorbar(sc, ax=ax, label="Nombre d'évaluations")
            ax.set_title(f"K={K}", fontsize=13)
            ax.set_xlabel("Fitness réelle", fontsize=11)
            ax.set_ylabel("Fitness prédite", fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()


def plot_covariate_shift(covariate_data_by_K, subdir=None, tag=None):
    sorted_K = sorted(covariate_data_by_K.keys())
    with figure("covariate_shift", tag, subdir=subdir):
        fig, axes = plt.subplots(1, len(sorted_K), figsize=(7 * len(sorted_K), 5), squeeze=False)

        for ax, K in zip(axes[0], sorted_K):
            dist_data = covariate_data_by_K[K]
            distances = sorted(dist_data.keys())
            means = [np.mean(dist_data[d]) for d in distances]
            stds  = [np.std(dist_data[d])  for d in distances]

            color = K_COLORS.get(K, 'black')
            ax.plot(distances, means, 'o-', color=color, linewidth=2, label=f'K={K}')
            ax.fill_between(distances,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color=color)

            ax.set_xlabel("Distance de Hamming à best_sol", fontsize=11)
            ax.set_ylabel("MAE", fontsize=11)
            ax.set_title(f"K={K}", fontsize=13)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
