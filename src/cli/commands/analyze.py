import numpy as np
from collections import defaultdict

from .base import Command
from src.utils.io_utils import build_plot_tag
from src.utils.analysis_report import analyse_optimization_file
from src.visualization import plot_aggregated_fidelity, plot_aggregated_convergence, plot_aggregated_fidelity_with_time, plot_confusion_metrics_distribution
from src.core.evaluation import compute_fidelity_mae, create_metrics_dataframe

class AnalyzeCommand(Command):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("path", type=str, nargs='+')

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
                    maes.append(compute_fidelity_mae(fidelity))

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
            plot_subdir, plot_tag = build_plot_tag(plot_N, K_str, plot_surrogate, plot_algo, plot_alpha, plot_tabu_factor)
        else:
            plot_subdir = None
            plot_tag = None
        self._report_results(all_matrices_by_K, N, maes, max_fitness_by_K, fidelity_samples_by_K, fitness_trajectories_by_K, final_scatter_by_K, mean_local_optima_by_K, plot_subdir, plot_tag, plot_surrogate, plot_algo)


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

def print_run_summary(filename, confusion_matrix, sparsity, N, K):

    s, z, t = sparsity

    print(f"\nMétriques - {filename} : {s:.1f}% zeros ({z}/{t})")

    print(create_metrics_dataframe(confusion_matrix, N, K))
