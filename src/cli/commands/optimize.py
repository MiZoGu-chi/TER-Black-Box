import os
from joblib import Parallel, delayed
from datetime import datetime
import multiprocessing

from .base import Command
from src.core.models import MODELS
from src.core.algorithm import make_baseline, ALL_ALGORITHM_NAMES, ALL_BASELINE_NAMES, get_algorithm_factory
from src.core.start_strategies import START_STRATEGIES
from src.core.surrogate_optimization import DUPLICATE_STRATEGIES
from src.utils.optimization_run import RunConfig, run_and_save, result_file_path, load_baselines
from src.utils.io_utils import OptimizationResult

class OptimizeCommand(Command):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--D", type=int, default=2)
        parser.add_argument("--N", type=int, nargs='+', default=[64], choices=[20, 64, 128, 256])
        parser.add_argument("--K", type=int, nargs='+', default=[1, 2], choices=[1, 2, 4, 8])
        parser.add_argument("--budget", type=int, default=1000)
        parser.add_argument("--initial_pop_size", type=int, default=20)
        parser.add_argument("--surrogate", type=str, default='lasso_lars', choices=MODELS.keys())
        parser.add_argument("--algo", type=str, default='cpp_tabu', choices=ALL_ALGORITHM_NAMES)
        parser.add_argument("--baseline", type=str, default='bi', choices=ALL_BASELINE_NAMES)
        parser.add_argument("--start", type=str, default='rd', choices=START_STRATEGIES.keys())
        parser.add_argument("--alpha", type=float, default=0.00001, help="alpha de lasso lars : terme de régularisation L1")
        parser.add_argument("--perturbation_strength", type=float, default=0.05, help="this (meta-)parameter is called the strength of the perturbation : drils perturbs the first parent by randomly flipping perturbation_strength * n bits")
        parser.add_argument("--tabu_factor", type=int, default=10)
        parser.add_argument("--drils_factor", type=int, default=10)
        parser.add_argument("--duplicate_strategy", type=str, default='random', choices=DUPLICATE_STRATEGIES)

        instance_selector = parser.add_mutually_exclusive_group()
        instance_selector.add_argument("--instances", type=int, nargs='+', choices=range(10), help="Pour tester des instances spécifiques (attention : réduire le nombre d'instances ne réduit pas le temps d'exécution tant que n_jobs > n_instances)")
        instance_selector.add_argument("--n_instances", type=int, default=10, choices=range(1, 11))

        seed_selector = parser.add_mutually_exclusive_group()
        seed_selector.add_argument("--seeds", type=int, nargs='+')
        seed_selector.add_argument("--n_seeds", type=int, default=10)

    def __init__(self, args):
        super().__init__(args)

        self.configs_N = args.N
        self.configs_K = args.K
        self.config_seeds = args.seeds or range(args.n_seeds)
        self.config_n_instances = args.instances or range(args.n_instances)

        model = MODELS[args.surrogate]['model']
        
        if 'alpha' in model.get_params():
            model.set_params(alpha=args.alpha)

        self.config = RunConfig(
            D=args.D,
            algo=args.algo,
            surrogate_name=args.surrogate,
            model=model,
            make_algorithm_fn=get_algorithm_factory(
                args.algo,
                tabu_factor=args.tabu_factor,
                perturbation_strength=args.perturbation_strength,
                drils_factor=args.drils_factor
            ),
            baseline_algo=make_baseline(args.baseline),
            start_fn=START_STRATEGIES[args.start],
            budget=args.budget,
            initial_pop_size=args.initial_pop_size,
            duplicate_strategy=args.duplicate_strategy,
            tabu_factor=args.tabu_factor,
            drils_factor=args.drils_factor,
        )

        self.n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    def execute(self):
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"

        baselines = load_baselines(self.configs_N, self.configs_K, self.config_n_instances, self.config.baseline_algo)

        Parallel(n_jobs=self.n_jobs)(delayed(run_and_save)(self.config, N, K, num, seed, timestamp, baselines[(N, K, num)])
            for N in self.configs_N
            for K in self.configs_K
            for num in self.config_n_instances
            for seed in self.config_seeds
        )

        result_files = sorted(
            result_file_path(self.config, N, K, num, seed, timestamp)
            for N in self.configs_N
            for K in self.configs_K
            for num in self.config_n_instances
            for seed in self.config_seeds
        )

        for f in result_files:
            try:
                res = OptimizationResult(f)
                s, z, t = res.sparsity
                print(f"[{f.name}] : {s:.1f}% zeros ({z}/{t})")

            except Exception:
                print(f"[{f.name}] : Erreur de Parsing")
