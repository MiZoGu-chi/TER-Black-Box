import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from .utils import figure, K_COLORS



def plot_aggregated_convergence(fitness_trajectories_by_K, mean_local_optima_by_K=None, subdir=None, tag=None):
    if not fitness_trajectories_by_K:
        return
    with figure("optimization_convergence", "trajectories", tag, subdir=subdir):
        plt.figure(figsize=(10, 6))
        for K, trajectories in fitness_trajectories_by_K.items():
            color = K_COLORS.get(K, 'black')
            eval_grid = np.array(sorted(set(item[1] for run in trajectories for item in run)))

            resampled = []
            for run in trajectories:
                evals = [item[1] for item in run]
                fitness = [item[0] for item in run]
                interp_fn = interp1d(evals, fitness, kind='previous',
                                     bounds_error=False, fill_value=(fitness[0], fitness[-1]))
                resampled.append(interp_fn(eval_grid))

            mean = np.mean(np.array(resampled), axis=0)
            plt.plot(eval_grid, mean, color=color, linewidth=2, label=f"K={K}")

            if mean_local_optima_by_K is not None and K in mean_local_optima_by_K:
                plt.axhline(y=np.mean(mean_local_optima_by_K[K]),
                            color=color, linestyle='--', linewidth=1.5, label=f"Moy. optima locaux (K={K})")

        plt.xlabel("Nombre d'évaluations", fontsize=12)
        plt.ylabel("Fitness", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

def plot_covariate_scatter(covariate_scatter_by_K, N, n_runs_by_K, subdir=None, tag=None):
    for K, dist_data in covariate_scatter_by_K.items():
        with figure(f"covariate_scatter_K{K}", tag, subdir=subdir):
            fig, ax = plt.subplots(figsize=(9, 7))

            distances = sorted(dist_data.keys())
            
            all_reals = []
            all_preds = []
            all_dists = []

            min_val = float('inf')
            max_val = float('-inf')

            for dist in distances:
                reals = dist_data[dist]['real']
                preds = dist_data[dist]['pred']
                if reals:
                    all_reals.extend(reals)
                    all_preds.extend(preds)
                    all_dists.extend([dist] * len(reals))
                    
                    min_val = min(min_val, min(reals), min(preds))
                    max_val = max(max_val, max(reals), max(preds))

            cmap_discrete = cm.get_cmap('plasma', len(distances))

            scatter = ax.scatter(all_reals, all_preds, c=all_dists, 
                                 cmap=cmap_discrete, alpha=0.6, s=15)

            if min_val != float('inf'):
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)

            cbar = fig.colorbar(scatter, ax=ax, ticks=distances)
            cbar.set_label("Distance de Hamming")

            ax.set_xlabel("Fitness réelle", fontsize=12)
            ax.set_ylabel("Fitness prédite", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()