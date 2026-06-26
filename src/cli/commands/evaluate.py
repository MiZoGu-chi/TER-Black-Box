from collections import defaultdict

from .base import Command
from src.utils.io_utils import OptimizationResult, build_plot_tag, load_landscape_and_masks
from src.core.sampling import SAMPLERS
from src.core.fitness import get_black_box_fitness_fn
from src.core.evaluation import load_fitted_model, evaluate_global_fidelity, evaluate_covariate_shift
from src.visualization import plot_aggregated_fidelity, plot_covariate_shift, plot_covariate_scatter

class EvaluateCommand(Command):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("path", type=str)
        parser.add_argument("--samples", type=int, default=1000)
        parser.add_argument("--sampler", type=str, default='whitebox', choices=list(SAMPLERS.keys()))

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

        # le sampler en whitebox est coûteux (recherche locale sur le vrai paysage). Plusieurs runs
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
                    plot_subdir, plot_tag = build_plot_tag(
                        N, K, meta['surrogate'], meta['algo'], meta.get('alpha'), meta.get('tabu_factor'))
                if plot_N is None:
                    plot_N = N

                model = load_fitted_model(meta['surrogate'], opti_res.coefs)
                landscape, masks = load_landscape_and_masks(N, K, D, num)
                fit_fn = get_black_box_fitness_fn(landscape)

                instance_key = (N, K, D, num)
                if instance_key not in test_solutions_cache:
                    test_solutions_cache[instance_key] = self.sampler.generate_samples(N, self.samples, seed=num, landscape=landscape)
                test_solutions = test_solutions_cache[instance_key]

                scatter = evaluate_global_fidelity(model, masks, fit_fn, test_solutions)
                final_scatter_by_K[K].append(scatter)

                if opti_res.best_sol is not None: # rétrocompatibilité avec les anciens fichiers
                    n_runs_by_K[K] += 1
                    covariate_data, covariate_scatter = evaluate_covariate_shift(
                        model, masks, fit_fn, opti_res.best_sol, run_seed, self.samples)

                    for dist, values in covariate_data.items():
                        covariate_data_by_K[K][dist].extend(values)
                    for dist, scat in covariate_scatter.items():
                        covariate_scatter_by_K[K][dist]['real'].extend(scat['real'])
                        covariate_scatter_by_K[K][dist]['pred'].extend(scat['pred'])

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
