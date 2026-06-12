import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

from .base import Command

from src.core.nk import load_landscape_and_masks

from src.utils.io_utils import OptimizationResult
from src.visualization import plot_aggregated_fidelity, plot_aggregated_convergence, plot_aggregated_fidelity_with_time, plot_confusion_metrics_distribution
from src.core.evaluation import get_surrogate_interaction, compare_interaction, compute_confusion_metrics

class AnalyzeCommand(Command):
    
    def execute(self):

        files = self._get_files()

        all_matrices_by_K = defaultdict(list)

        maes = []

        N = 0
        plot_N, plot_surrogate, plot_algo, plot_alpha, plot_tabu_factor = None, None, None, None, None
        all_K = set()

        max_fitness_by_K = defaultdict(list)
        fitness_trajectories_by_K = defaultdict(list)
        fidelity_samples_by_K = defaultdict(list)
        final_scatter_by_K = defaultdict(list)
        mean_local_optima_by_K = defaultdict(list)

        for file in files:
            try:
                meta, confusion_matrix, fidelity, fitness, final_scatter, sparsity = analyse_optimization_file(file)

                N, K, D = meta['N'], meta['K'], meta['D']
                all_K.add(K)

                if plot_N is None:
                    plot_N, plot_surrogate, plot_algo = N, meta['surrogate'], meta['algo']
                    plot_alpha, plot_tabu_factor = meta.get('alpha'), meta.get('tabu_factor')

                if meta.get('mean_local_optima') is not None:
                    mean_local_optima_by_K[K].append(meta['mean_local_optima'])

                all_matrices_by_K[K].append(confusion_matrix)

                print_run_summary(file.name, confusion_matrix, sparsity, N, K)

                if len(fidelity) > 0:
                    fidelity_samples_by_K[K].extend(fidelity)
                    maes.append(self._get_mae(fidelity))

                if len(final_scatter) > 0:
                    final_scatter_by_K[K].append(final_scatter)

                if len(fitness) > 0:
                    fitness_trajectories_by_K[K].append(fitness)

                    last_score = fitness[-1][0]
                    max_fitness_by_K[K].append(last_score)

            except Exception as e:
                print(f"Erreur lors de l'analyse de {file.name} : {e}")

        K_str = "_".join(str(k) for k in sorted(all_K))
        if plot_N:
            plot_subdir = f"{plot_N}/{K_str}/{plot_surrogate}"
            if plot_alpha is not None:
                plot_subdir += f"/{plot_alpha:g}"
            plot_subdir += f"/{plot_algo}"
            if plot_tabu_factor is not None and 'tabu' in plot_algo:
                plot_subdir += f"/{plot_tabu_factor}"
            plot_tag = plot_subdir.replace('/', '_')
        else:
            plot_subdir = None
            plot_tag = None
        self._report_results(all_matrices_by_K, N, maes, max_fitness_by_K, fidelity_samples_by_K, fitness_trajectories_by_K, final_scatter_by_K, mean_local_optima_by_K, plot_subdir, plot_tag, plot_surrogate, plot_algo)


    def _get_mae(self, fidelity_history):
        preds = [p[0] for p in fidelity_history]
        reals = [p[1] for p in fidelity_history]
        return mean_absolute_error(reals, preds)

    def _report_results(self, all_matrices_by_K, N, maes, max_fitness_by_K, fidelity_samples_by_K, fitness_trajectories_by_K, final_scatter_by_K, mean_local_optima_by_K, subdir=None, tag=None, surrogate=None, algo=None):

        for K, scores in sorted(max_fitness_by_K.items()):
            mean_score = np.mean(scores)

            print(f'Score agrégé sur {len(scores)} runs pour K={K}: {mean_score:.4f}')
            print(f'Ecart-type sur {len(scores)} runs pour K={K}: {np.std(scores):.4f}')

            if K in mean_local_optima_by_K and len(mean_local_optima_by_K[K]) > 0:
                mean_exp = np.mean(mean_local_optima_by_K[K])
                print(f'Moy. optima locaux sur {len(scores)} runs pour K={K}: {mean_exp:.4f}')
                print(f'Amélioration sur {len(scores)} runs pour K={K}: {mean_score - mean_exp:+.4f}\n')
            else:
                print(f'Moy. optima locaux sur {len(scores)} runs pour K={K}: Non disponible (ancienne run)\n')

        if not any(mean_local_optima_by_K.values()):
            mean_local_optima_by_K = None

        plot_aggregated_convergence(fitness_trajectories_by_K, mean_local_optima_by_K, subdir=subdir, tag=tag)
        plot_aggregated_fidelity_with_time(fidelity_samples_by_K, subdir=subdir, tag=tag)
        plot_confusion_metrics_distribution(all_matrices_by_K, N, surrogate=surrogate, algo=algo, subdir=subdir, tag=tag)

##########

def analyse_optimization_file(file):

    opti_res = OptimizationResult(file)
    
    s, z, t = opti_res.sparsity

    N = opti_res.meta['N']
    K = opti_res.meta['K']
    D = opti_res.meta['D']
    num = opti_res.meta['num'] 
    seed = opti_res.meta['seed']

    landscape, masks = load_landscape_and_masks(N, K, D, num)

    # Format : { ordre1 : [((tuple1), coef1), ..., ((tupleN), coefN)], ordreN: ... }
    # Interactions + poids
    surrogate_graph = get_surrogate_interaction(opti_res.coefs, masks)
        
    # Format : { ordre1 : [(tuple1), ..., (tupleN)], ordreN: ... }
    # Uniquement les intéractions 
    real_graph = landscape.get_real_interaction()

    confusion_matrix_by_K = compare_interaction(surrogate_graph, real_graph, D)

    return opti_res.meta, confusion_matrix_by_K, opti_res.fidelity_history, opti_res.fitness_history, opti_res.final_scatter, opti_res.sparsity


def print_run_summary(filename, confusion_matrix, sparsity, N, K):       

    s, z, t = sparsity

    print(f"\nMétriques - {filename} : {s:.1f}% zeros ({z}/{t})")

    print(create_metrics_dataframe(confusion_matrix, N, K))


def create_metrics_dataframe(confusion_matrix_by_order, N, K):
    tp, fp, fn, precision, recall, f1 = compute_confusion_metrics(confusion_matrix_by_order)
    
    df = pd.DataFrame({
        'N': [N] * len(tp),
        'K': [K] * len(tp),
        'D': range(1, len(tp) + 1),
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall, 
        'F1': f1
    })
    
    for col in ['Precision', 'Recall', 'F1']:
        df[col] = df[col].map("{:.2%}".format) 
                  
    return df