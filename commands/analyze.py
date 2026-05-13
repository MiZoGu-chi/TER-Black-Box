import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

from .base import Command  
from walsh import generate_masks
from io_utils import OptimizationResult
from nk import NK_Landscape
from plot import plot_optimization, plot_aggregated_fidelity, plot_aggregated_fitness_distributions, plot_aggregated_convergence, save_df_as_image
from evaluation import get_surrogate_interaction, compare_interaction, compute_confusion_metrics

class AnalyzeCommand(Command):
    # crash sur des folders avec des run sur des N / D différents 
    def execute(self):

        files = self._get_files()

        agreged_matrix = None

        maes = []

        max_fitness_by_K = defaultdict(list)
        fitness_trajectories_by_K = defaultdict(list)
        fidelity_samples_by_K = defaultdict(list)

        for file in files:
            try:
                meta, confusion_matrix, fidelity, fitness, sparsity = analyse_optimization_file(file)

                N, K = meta['N'], meta['K']

                if agreged_matrix is None: 
                    agreged_matrix = np.zeros_like(confusion_matrix)
                agreged_matrix += confusion_matrix

                print_run_summary(file.name, confusion_matrix, sparsity, N)       

                if fidelity:
                    fidelity_samples_by_K[K].extend(fidelity)

                    maes.append(self._get_mae(fidelity))

                if fitness:
                    fitness_trajectories_by_K[K].append(fitness)

                    last_score = fitness[-1][0]
                    max_fitness_by_K[K].append(last_score)
                
            except Exception as e:
                print(f"Erreur lors de l'analyse de {file.name} : {e}")

        df_metrics = create_metrics_dataframe(agreged_matrix, N)
        #print(create_metrics_dataframe(agreged_matrix))

        print(f'mae surrogate : {np.mean(maes)}')

        for K, scores in sorted(max_fitness_by_K.items()):
            print(f'Score agrégé sur {len(scores)} runs pour K={K}: {np.mean(scores):.4f}')
            print(f'Ecart-type sur {len(scores)} runs pour K={K}: {np.std(scores):.4f}')
        
        plot_aggregated_fitness_distributions(max_fitness_by_K)
        plot_aggregated_convergence(fitness_trajectories_by_K)
        plot_aggregated_fidelity(fidelity_samples_by_K)
        save_df_as_image(df_metrics)

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

    def _report_results(self, matrix, maes, max_fit, trajectories, fidelity_samples):

        df_metrics = create_metrics_dataframe(matrix)
        print(f'\nmae surrogate : {np.mean(maes)}')
        
        for K, scores in sorted(max_fit.items()):
            print(f'Score agrégé sur {len(scores)} runs pour K={K}: {np.mean(scores):.4f}')
            print(f'Ecart-type sur {len(scores)} runs pour K={K}: {np.std(scores):.4f}')
        
        plot_aggregated_fitness_distributions(max_fit)
        plot_aggregated_convergence(trajectories)
        plot_aggregated_fidelity(fidelity_samples)
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

    return opti_res.meta, confusion_matrix_by_K, opti_res.fidelity_history, opti_res.fitness_history, opti_res.sparsity

def print_run_summary(filename, confusion_matrix, sparsity, N):       

    s, z, t = sparsity

    print(f"\nMétriques - {filename} : {s:.1f}% zeros ({z}/{t})")

    print(create_metrics_dataframe(confusion_matrix, N))

def create_metrics_dataframe(confusion_matrix_by_order, N):
    tp, fp, fn, precision, recall, f1 = compute_confusion_metrics(confusion_matrix_by_order)
    
    df = pd.DataFrame({
        'N': [N] * len(tp),
        'K': range(1, len(tp) + 1),
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