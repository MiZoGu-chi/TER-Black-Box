import numpy as np
from collections import defaultdict
from sklearn.base import clone

from .base import Command
from src.utils.io_utils import OptimizationResult
from src.core.data_sampler import transform
from src.core.nk import load_landscape_and_masks
from src.core.models import MODELS
from src.core.sampling import RandomSampler, StratifiedSampler, WhiteBoxStratifiedSampler, generate_hamming_neighbors, SAMPLERS
from src.core.algorithm import get_black_box_fitness_fn
from src.visualization import plot_aggregated_fidelity, plot_covariate_shift, plot_covariate_scatter

def _distance_grid(N, n_points):
    return np.unique(np.round(np.linspace(0, N // 2, n_points)).astype(int))

class EvaluateCommand(Command):

    def __init__(self, args):
        super().__init__(args)
        self.sampler = SAMPLERS[args.sampler]()
        self.samples = args.samples

    def execute(self):
        files = self._get_files()

        if not files:
            print("Aucun fichier trouvé.")
            return

        print(f"Lancement de l'évaluation sur {len(files)} runs...")


        # le samplet en whitebox est coûteux (recherche locale sur le vrai paysage). Plusieurs runs
        # sur la même instance (N, K, D, num) ne diffèrent que par la seed d'optimisation et non par
        # le paysage, on recalcule l'ensemble de test une seule fois par instance.
        test_solutions_cache = {}

        final_scatter_by_K = defaultdict(list)
        covariate_data_by_K = defaultdict(lambda: defaultdict(list))
        covariate_scatter_by_K = defaultdict(lambda: defaultdict(lambda: {'real': [], 'pred': []}))
        n_runs_by_K = defaultdict(int)
        
        plot_subdir = None
        plot_tag = None
        plot_N = None

        for file in files:
            try:
                opti_res = OptimizationResult(file)
                meta = opti_res.meta
                N, K, D, num, run_seed = meta['N'], meta['K'], meta['D'], meta['num'], meta['seed']

                if plot_subdir is None:
                    alpha = meta.get('alpha')
                    tabu_factor = meta.get('tabu_factor')
                    plot_subdir = f"{N}/{K}/{meta['surrogate']}"
                    if alpha is not None:
                        plot_subdir += f"/{alpha:g}"
                    plot_subdir += f"/{meta['algo']}"
                    if tabu_factor is not None and 'tabu' in meta['algo']:
                        plot_subdir += f"/{tabu_factor}"
                    plot_tag = plot_subdir.replace('/', '_')
                if plot_N is None:
                    plot_N = N

                model = clone(MODELS[meta['surrogate']]['model'])
                model.coef_ = np.array(opti_res.coefs)
                model.intercept_ = 0.0

                landscape, masks = load_landscape_and_masks(N, K, D, num)
                fit_fn = get_black_box_fitness_fn(landscape)

                # Fidélité sur des points qui ne correspondent pas forcément aux zones explorée
                instance_key = (N, K, D, num)
                if instance_key not in test_solutions_cache:
                    test_solutions_cache[instance_key] = self.sampler.generate_samples(N, self.samples, seed=num, landscape=landscape)
                test_solutions = test_solutions_cache[instance_key]
                real_fitness_global = fit_fn(test_solutions)
                pred_fitness_global = model.predict(transform(masks, test_solutions))

                scatter = np.column_stack([pred_fitness_global, real_fitness_global])
                final_scatter_by_K[K].append(scatter)

                # Fidélité en s'éloignant petit à petit de la meilleur solution trouvée
                if opti_res.best_sol is not None: # rétrocompatibilité avec les anciens fichiers
                    n_runs_by_K[K] += 1
                    rng = np.random.default_rng(run_seed) # attention on a déjà fait une fonction pour la rng, à remplacer

                    for dist in _distance_grid(N, 9): 
                        sols = generate_hamming_neighbors(opti_res.best_sol, int(dist), self.samples // 10, rng) # casting inutile ?
                        
                        real_f = fit_fn(sols)
                        pred_f = model.predict(transform(masks, sols))
                        
                        covariate_data_by_K[K][int(dist)].extend(np.abs(pred_f - real_f).tolist())
                        covariate_scatter_by_K[K][int(dist)]['real'].extend(real_f.tolist())
                        covariate_scatter_by_K[K][int(dist)]['pred'].extend(pred_f.tolist())

            except Exception as e:
                print(f"Erreur lors de l'évaluation de {file.name} : {e}")

        print("Génération du graphique de fidélité globale...")
        plot_aggregated_fidelity(final_scatter_by_K, subdir=plot_subdir, tag=plot_tag)

        if any(v for v in covariate_data_by_K.values()):
            print("Génération des graphiques Covariate Shift...")
            plot_covariate_shift(covariate_data_by_K, subdir=plot_subdir, tag=plot_tag)
            plot_covariate_scatter(covariate_scatter_by_K, plot_N, n_runs_by_K, subdir=plot_subdir, tag=plot_tag)
        else:
            print("Aucune best_sol trouvée dans ces fichiers. Le graphique de Covariate Shift ne peut pas être généré.")
