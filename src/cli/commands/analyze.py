import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

from .base import Command  

from src.core.walsh import generate_masks
from src.core.nk import NK_Landscape

from src.utils.io_utils import OptimizationResult
from src.utils.plot import plot_aggregated_fidelity, plot_aggregated_fitness_distributions, plot_aggregated_convergence, save_df_as_image, plot_aggregated_fidelity_with_time
from src.core.evaluation import get_surrogate_interaction, compare_interaction, compute_confusion_metrics

class AnalyzeCommand(Command):
    
    def execute(self):

        files = self._get_files()

        agreged_matrix = {}

        maes = []

        N = 0

        max_fitness_by_K = defaultdict(list)
        fitness_trajectories_by_K = defaultdict(list)
        fidelity_samples_by_K = defaultdict(list)
        final_scatter_by_K = defaultdict(list)
        expected_local_optima_by_K = defaultdict(list)

        for file in files:
            try:
                meta, confusion_matrix, fidelity, fitness, final_scatter, sparsity = analyse_optimization_file(file)

                N, K, D = meta['N'], meta['K'], meta['D']

                expected_local_optima_by_K[K].append(meta['expected_local_optima'])

                if K not in agreged_matrix:
                    agreged_matrix[K] = np.zeros_like(confusion_matrix)
                agreged_matrix[K] += confusion_matrix

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

        self._report_results(agreged_matrix, N, maes, max_fitness_by_K, fidelity_samples_by_K, fitness_trajectories_by_K, final_scatter_by_K, expected_local_optima_by_K) 

    def _get_files(self):
        path = Path(self.args.path)
        if path.is_dir():
            return sorted(path.glob("opti_*.txt"))
        if path.is_file():
            return [path]
        return []

    def _get_mae(self, fidelity_history):
        preds = [p[0] for p in fidelity_history]
        reals = [p[1] for p in fidelity_history]
        return mean_absolute_error(reals, preds)

    def _report_results(self, agreged_matrix, N, maes, max_fitness_by_K, fidelity_samples_by_K, fitness_trajectories_by_K, final_scatter_by_K, expected_local_optima_by_K):

        # concaténation des matrices de confusions
        df_metrics = pd.concat([
            create_metrics_dataframe(m, N, k)
            for k, m in sorted(agreged_matrix.items())
        ], ignore_index=True)

        for K, scores in sorted(max_fitness_by_K.items()):
            mean_score = np.mean(scores)
            mean_exp = np.mean(expected_local_optima_by_K[K])
 
            print(f'Score agrégé sur {len(scores)} runs pour K={K}: {mean_score:.4f}')
            print(f'Ecart-type sur {len(scores)} runs pour K={K}: {np.std(scores):.4f}')
            print(f'Espérance sur {len(scores)} runs pour K={K}: {mean_exp:.4f}')
            print(f'Amélioration sur {len(scores)} runs pour K={K}: {mean_score - mean_exp:+.4f}\n')

        plot_aggregated_fitness_distributions(max_fitness_by_K, expected_local_optima_by_K)
        plot_aggregated_convergence(fitness_trajectories_by_K, expected_local_optima_by_K)
        plot_aggregated_fidelity_with_time(fidelity_samples_by_K)
        plot_aggregated_fidelity(final_scatter_by_K)
        save_df_as_image(df_metrics)

##########

def analyse_optimization_file(file):

    opti_res = OptimizationResult(file)
    
    s, z, t = opti_res.sparsity

    N = opti_res.meta['N']
    K = opti_res.meta['K']
    D = opti_res.meta['D']
    num = opti_res.meta['num'] 
    seed = opti_res.meta['seed']

    masks = generate_masks(N, D)

    # Format : { ordre1 : [((tuple1), coef1), ..., ((tupleN), coefN)], ordreN: ... }
    # Interactions + poids
    surrogate_graph = get_surrogate_interaction(opti_res.coefs, masks)

    # On est obligé de recréer l'instance NK
    instance_name = f"instances/{N}/{K}/nk_{N}_{K}_{num}.txt"
    landscape = NK_Landscape(instance_name)
        
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