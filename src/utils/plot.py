import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os

import numpy as np

from scipy.interpolate import interp1d

from src.utils.io_utils import OptimizationResult
 
def plot_compare_models(models, D, instance_name, loss_function):
    plt.figure(figsize=(10, 6))

    for m, info in models.items():
        plt.plot(info['loss_values'], linewidth=2.0, alpha=0.8, label=info['name'], color=info['color'])

    plt.title(f"Comparaison des modèles (D={D})\nInstance : {instance_name}", fontsize=16)

    plt.xlabel("Sample size", fontsize=12)
    plt.ylabel("Mean Abs. Error of fitness", fontsize=12)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    plt.legend(title='method', loc='upper right', fontsize=12)
    
    plt.tight_layout()

    save_plot(instance_name, f"D={D}", loss_function)



def plot_aggregated_fitness_distributions(max_fitness_by_K, expected_fitness_by_K=None):
    plt.figure(figsize=(8, 6))
    
    sorted_K = sorted(max_fitness_by_K.keys())
    fitness_distributions = [max_fitness_by_K[k] for k in sorted_K]
    
    plt.boxplot(fitness_distributions, labels=[f"K={k}" for k in sorted_K])
    
    # ajout de l'espérance sous forme d'étoile rouge
    if expected_fitness_by_K is not None:
        for i, k in enumerate(sorted_K):
            if k in expected_fitness_by_K:
                mean_exp = np.mean(expected_fitness_by_K[k])
                plt.plot(i + 1, mean_exp, marker='*', color='red', markersize=10, linestyle='None', label='Espérance' if i == 0 else "")
    
    plt.title("Distribution de la fitness par valeur de K", fontsize=14)
    plt.ylabel("Fitness", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if expected_fitness_by_K is not None:
        plt.legend()
        
    save_plot("aggregated_fitness")
    plt.close()    

def plot_aggregated_convergence(fitness_trajectories_by_K, expected_local_optima_by_K=None):
    plt.figure(figsize=(10, 6))
    colors = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red'}

    for K, trajectories in fitness_trajectories_by_K.items():
        color = colors.get(K, 'black')

        # union de tous les points d'évaluation réels pour ce K
        eval_grid = np.array(sorted(set(item[1] for run in trajectories for item in run)))

        resampled = []
        for run in trajectories:
            evals = [item[1] for item in run]
            fitness = [item[0] for item in run]
            # à chaque point de la grille, dernière valeur connue
            interp_fn = interp1d(evals, fitness, kind='previous', bounds_error=False, fill_value=(fitness[0], fitness[-1]))
            resampled.append(interp_fn(eval_grid))

        resampled = np.array(resampled)
        mean = np.mean(resampled, axis=0)

        plt.plot(eval_grid, mean, color=color, linewidth=2, label=f"K={K}")

        if expected_local_optima_by_K is not None and K in expected_local_optima_by_K:
            mean_exp = np.mean(expected_local_optima_by_K[K])
            plt.axhline(y=mean_exp, color=color, linestyle='--', linewidth=1.5, label=f"Espérance (K={K})")

    plt.title("Convergence agrégée par K", fontsize=14)
    plt.xlabel("Nombre d'évaluations", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    save_plot("optimization_convergence", "trajectories")
    plt.close()

def plot_aggregated_fidelity_with_time(fidelity_samples_by_K):
    sorted_K = sorted(fidelity_samples_by_K.keys())
    fig, axes = plt.subplots(1, len(sorted_K), figsize=(7 * len(sorted_K), 6))

    # Si un seul K, axes n'est pas une liste
    if len(sorted_K) == 1:
        axes = [axes]

    for ax, K in zip(axes, sorted_K):
        samples = fidelity_samples_by_K[K]
        preds  = [s[0] for s in samples]
        reals  = [s[1] for s in samples]
        n_evals = [s[2] for s in samples]

        sc = ax.scatter(reals, preds, c=n_evals, cmap='viridis', s=8, alpha=0.6)

        # Ligne diagonale de référence
        min_val = min(min(reals), min(preds))
        max_val = max(max(reals), max(preds))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        fig.colorbar(sc, ax=ax, label="Nombre d'évaluations")
        ax.set_title(f"K={K}", fontsize=13)
        ax.set_xlabel("Fitness réelle", fontsize=11)
        ax.set_ylabel("Fitness prédite", fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Évolution de la précision du surrogate pendant l'optimisation", fontsize=14)
    plt.tight_layout()
    save_plot("model_training_fidelity")
    plt.close()

def plot_aggregated_fidelity(final_scatter_by_K):
    plt.figure(figsize=(8, 8))
    colors = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red'}

    all_preds, all_reals = [], []

    for K, scatter_list in final_scatter_by_K.items():
        # scatter_list est une liste de tableaux (n_samples, 2) — un par run
        scatter = np.vstack(scatter_list)
        preds = scatter[:, 0]
        reals = scatter[:, 1]

        all_preds.extend(preds)
        all_reals.extend(reals)

        plt.scatter(reals, preds, s=5, color=colors.get(K, 'black'), alpha=0.3, label=f"K={K}")

    if all_preds:
        min_val = min(min(all_reals), min(all_preds))
        max_val = max(max(all_reals), max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'b--')

    plt.title("Précision globale du modèle surrogate par K", fontsize=14)
    plt.xlabel("Fitness réelle", fontsize=12)
    plt.ylabel("Fitness prédite", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("model_fidelity")
    plt.close()
    
def save_plot(base_name, *tags):
    filename = base_name

    for tag in tags:
        if tag:
            filename += f"_{tag}"
            
    extension = ".png"
    path = f"results/plots/{filename}{extension}"
    
    counter = 1
    while os.path.exists(path):
        path = f"results/plots/{filename}_{counter}{extension}"
        counter += 1
    
    plt.savefig(path)
    print(f"Graphique sauvegardé : {path}")