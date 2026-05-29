import numpy as np
from collections import defaultdict
from sklearn.base import clone

from .base import Command
from src.utils.io_utils import OptimizationResult
from src.core.data_sampler import transform
from src.core.nk import load_landscape_and_masks
from src.core.models import MODELS
from src.core.sampling import WhiteBoxStratifiedSampler
from src.core.algorithm import get_black_box_fitness_fn
from src.visualization import plot_aggregated_fidelity, plot_covariate_shift


class EvaluateCommand(Command):

    def __init__(self, args):
        super().__init__(args)
        self.sampler = WhiteBoxStratifiedSampler()

    def execute(self):
        files = self._get_files()
        if not files:
            print("Aucun fichier trouvé.")
            return

        print(f"Lancement de l'évaluation sur {len(files)} runs...")
        final_scatter_by_K = defaultdict(list)
        plot_subdir = None

        for file in files:
            try:
                opti_res = OptimizationResult(file)
                meta = opti_res.meta
                N, K, D, num, run_seed = meta['N'], meta['K'], meta['D'], meta['num'], meta['seed']

                if plot_subdir is None:
                    plot_subdir = f"{N}/{K}/{meta['surrogate']}/{meta['algo']}"

                model = clone(MODELS[meta['surrogate']]['model'])
                model.coef_ = np.array(opti_res.coefs)
                model.intercept_ = 0.0

                landscape, masks = load_landscape_and_masks(N, K, D, num)

                test_solutions = self.sampler.generate_samples(N, self.args.samples, seed=run_seed, landscape=landscape)
                real_fitness = get_black_box_fitness_fn(landscape)(test_solutions)

                X_walsh = transform(masks, test_solutions)
                pred_fitness = model.predict(X_walsh)

                scatter = np.column_stack([pred_fitness, real_fitness])
                final_scatter_by_K[K].append(scatter)

            except Exception as e:
                print(f"Erreur lors de l'évaluation de {file.name} : {e}")

        print("Génération du graphique de fidélité globale...")
        plot_aggregated_fidelity(final_scatter_by_K, subdir=plot_subdir)

