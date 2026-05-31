import os
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from scipy.interpolate import interp1d

K_COLORS = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red'}

RESULTS_DIR = "results/plots"


def save_plot(base_name, *tags, subdir=None):
    from pathlib import Path
    plot_dir = Path(RESULTS_DIR) / subdir if subdir else Path(RESULTS_DIR)
    plot_dir.mkdir(parents=True, exist_ok=True)

    filename = base_name
    for tag in tags:
        if tag:
            filename += f"_{tag}"

    path = plot_dir / f"{filename}.png"
    counter = 1
    while path.exists():
        path = plot_dir / f"{filename}_{counter}.png"
        counter += 1

    plt.savefig(path)
    print(f"Graphique sauvegardé : {path}")


@contextmanager
def _figure(filename, *tags, subdir=None):
    try:
        yield
    finally:
        save_plot(filename, *tags, subdir=subdir)
        plt.close()


def plot_compare_models(models, D, instance_name, loss_function):
    with _figure(instance_name, f"D={D}", loss_function):
        plt.figure(figsize=(10, 6))
        for m, info in models.items():
            plt.plot(info['loss_values'], linewidth=2.0, alpha=0.8, label=info['name'], color=info['color'])
        plt.title(f"Comparaison des modèles (D={D})\nInstance : {instance_name}", fontsize=16)
        plt.xlabel("Sample size", fontsize=12)
        plt.ylabel("Mean Abs. Error of fitness", fontsize=12)
        plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
        plt.legend(title='method', loc='upper right', fontsize=12)
        plt.tight_layout()


def plot_aggregated_fitness_distributions(max_fitness_by_K, expected_fitness_by_K=None, subdir=None):
    sorted_K = [k for k in sorted(max_fitness_by_K.keys()) if len(max_fitness_by_K[k]) > 0]
    if not sorted_K:
        return
    with _figure("aggregated_fitness", subdir=subdir):
        plt.figure(figsize=(8, 6))
        plt.boxplot([max_fitness_by_K[k] for k in sorted_K], labels=[f"K={k}" for k in sorted_K])

        if expected_fitness_by_K is not None:
            for i, k in enumerate(sorted_K):
                if k in expected_fitness_by_K:
                    plt.plot(i + 1, np.mean(expected_fitness_by_K[k]),
                             marker='*', color='red', markersize=10,
                             linestyle='None', label='Moy. optima locaux' if i == 0 else "")
            plt.legend()

        plt.title("Distribution de la fitness par valeur de K", fontsize=14)
        plt.ylabel("Fitness", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)


def plot_aggregated_convergence(fitness_trajectories_by_K, mean_local_optima_by_K=None, subdir=None):
    if not fitness_trajectories_by_K:
        return
    with _figure("optimization_convergence", "trajectories", subdir=subdir):
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

        plt.title("Convergence agrégée par K", fontsize=14)
        plt.xlabel("Nombre d'évaluations", fontsize=12)
        plt.ylabel("Fitness", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()


def plot_aggregated_fidelity_with_time(fidelity_samples_by_K, subdir=None):
    sorted_K = [k for k in sorted(fidelity_samples_by_K.keys()) if len(fidelity_samples_by_K[k]) > 0]
    if not sorted_K:
        return
    with _figure("model_training_fidelity", subdir=subdir):
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

        fig.suptitle("Évolution de la précision du surrogate pendant l'optimisation", fontsize=14)
        plt.tight_layout()


def plot_aggregated_fidelity(final_scatter_by_K, subdir=None):
    with _figure("model_fidelity", subdir=subdir):
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

        plt.title("Précision globale du modèle surrogate par K", fontsize=14)
        plt.xlabel("Fitness réelle", fontsize=12)
        plt.ylabel("Fitness prédite", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)


def plot_covariate_shift(covariate_data_by_K, subdir=None):
    """MAE = f(distance de Hamming à best_sol) — démontre l'expertise locale du surrogate."""
    sorted_K = sorted(covariate_data_by_K.keys())
    with _figure("covariate_shift", subdir=subdir):
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


def save_df_as_image(df, filename="interaction_metrics", subdir=None):
    with _figure(filename, subdir=subdir):
        fig, ax = plt.subplots(figsize=(10, 0.5 * (len(df) + 1)))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns,
                         rowLabels=df.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))


def plot_confusion_metrics_distribution(all_matrices_by_K, N, surrogate=None, algo=None, subdir=None):
    from src.core.evaluation import compute_confusion_metrics
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    sorted_K = sorted(all_matrices_by_K.keys())

    with _figure("confusion_distribution", subdir=subdir):
        fig, axes = plt.subplots(1, len(sorted_K), figsize=(6 * len(sorted_K), 5), squeeze=False)

        for ax, K in zip(axes[0], sorted_K):
            matrices = all_matrices_by_K[K]
            n_runs = len(matrices)
            D = matrices[0].shape[0]

            all_tp = np.zeros((n_runs, D))
            all_fp = np.zeros((n_runs, D))
            all_fn = np.zeros((n_runs, D))

            for i, m in enumerate(matrices):
                tp, fp, fn, _, _, _ = compute_confusion_metrics(m)
                all_tp[i] = tp
                all_fp[i] = fp
                all_fn[i] = fn

            positions, data, colors = [], [], []
            x_tick_pos, x_tick_labels = [], []

            for d in range(D):
                base = d * 3
                positions += [base, base + 1]
                data += [all_tp[:, d], all_fp[:, d]]
                colors += ['#2ecc71', '#e74c3c']
                x_tick_pos.append(base + 0.5)
                x_tick_labels.append(f"Ordre {d + 1}")

            bp = ax.boxplot(data, positions=positions, widths=0.6,
                            patch_artist=True, showfliers=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            for d in range(D):
                total_real = int(np.median(all_tp[:, d] + all_fn[:, d]))
                ax.plot(d * 3, total_real, marker='*', color='red',
                        markersize=13, zorder=5, linestyle='None')

            ax.set_xticks(x_tick_pos)
            ax.set_xticklabels(x_tick_labels)
            ax.set_ylabel("Nombre d'interactions", fontsize=11)
            info = "  ".join(filter(None, [surrogate, algo]))
            ax.set_title(f"K={K}  N={N}  {info}  {n_runs} runs", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            legend_elements = [
                Patch(facecolor='#2ecc71', alpha=0.6, label='TP'),
                Patch(facecolor='#e74c3c', alpha=0.6, label='FP'),
                Line2D([0], [0], marker='*', color='red', markersize=10,
                       linestyle='None', label='Total réel'),
            ]
            ax.legend(handles=legend_elements, fontsize=9)

        fig.suptitle(
            "Comparaison des interactions détectées par le surrogate\n"
            "avec les interactions réelles du paysage NK",
            fontsize=13
        )
        plt.tight_layout()
