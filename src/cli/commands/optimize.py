import os
from joblib import Parallel, delayed
from datetime import datetime
import numpy as np
import random
import multiprocessing
from pathlib import Path

from .base import Command
from src.core.nk import NK_Landscape, load_landscape_and_masks
from src.core.surrogate_optimization import run_surrogate_optimization
from src.core.models import MODELS
from src.core.algorithm import ALGORITHMS, BASELINE_ALGORITHMS, START_STRATEGIES, get_algorithm

from src.utils.io_utils import OptimizationResult

# TODO: extraire un dataclass RunConfig (budget, model, search_fn, ...) et faire de _run_instance une fonction standalone (Parameter Object pattern / SRP)
class OptimizeCommand(Command):

    def __init__(self, args):
        super().__init__(args)

        self.configs_N = args.N
        self.configs_K = args.K
        self.configs_D = args.D

        self.config_seeds = args.seeds or range(args.n_seeds)
        self.config_n_instances = args.instances or range(args.n_instances)

        self.algo = args.algo
        self.surrogate_name = args.surrogate
        self.model = MODELS[args.surrogate]['model']

        if 'alpha' in self.model.get_params():
            self.model.set_params(alpha=args.alpha)

        self.budget = args.budget
        self.initial_pop_size = args.initial_pop_size
        self.start_fn = START_STRATEGIES[args.start]
        self.tabu_factor = args.tabu_factor
        self.search_fn = get_algorithm(args.algo, tabu_factor=args.tabu_factor)
        self.n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    def _run_instance(self, N, K, num, seed, timestamp, mean_local_optima):

        np.random.seed(seed)
        random.seed(seed)

        landscape, masks = load_landscape_and_masks(N, K, self.configs_D, num)

        best_fit, best_sol, history, surrogate_model, surrogate_fidelity_history = run_surrogate_optimization(
            self.budget, N, self.initial_pop_size, landscape, masks, self.model, self.search_fn, self.start_fn)

        coefs = surrogate_model.coef_.tolist()

        output_dir = Path(f"results/opti_output/N{N}_K{K}_D{self.configs_D}_{self.surrogate_name}/run_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"opti_{N}_{K}_{num}_{seed}_{self.algo}_{self.surrogate_name}.txt"

        with open(filename, "w") as f:
            f.write(f"{N} {K} {self.configs_D} {num} {seed} {self.algo} {self.surrogate_name} {mean_local_optima} {self.model.alpha} {self.tabu_factor}\n")

            f.write("#\n")
            for sol, budg in history:
                f.write(f"{sol} {budg}\n")

            f.write("#\n")
            for pred, real, n_eval in surrogate_fidelity_history: # ajout coloration des points en fonction de n_evals
                f.write(f"{pred} {real} {n_eval}\n")

            f.write("#\n")
            f.write(f"{best_sol.tolist()}\n")

            f.write("#\n")
            f.write(f"{coefs}\n")

    def execute(self):
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"

        baseline_algo = 'bi' if self.algo.startswith('cpp_') else self.algo
        search_fn = BASELINE_ALGORITHMS[baseline_algo]

        baselines = {}

        for N in self.configs_N:
            for K in self.configs_K:
                for num in self.config_n_instances:
                    instance_name = f"instances/{N}/{K}/nk_{N}_{K}_{num}.txt"
                    if os.path.exists(instance_name):
                        landscape = NK_Landscape(instance_name)
                        baselines[(N, K, num)] = landscape.mean_local_optima(search_fn)

        Parallel(n_jobs=self.n_jobs)(delayed(self._run_instance)(N, K, num, seed, timestamp, baselines[(N, K, num)])
            for N in self.configs_N
            for K in self.configs_K
            for num in self.config_n_instances
            for seed in self.config_seeds
        )

        result_files = sorted(Path("results/opti_output").rglob(f"*run_{timestamp}*/opti_*.txt"))

        for f in result_files:
            try:
                res = OptimizationResult(f)
                s, z, t = res.sparsity
                print(f"[{f.name}] : {s:.1f}% zeros ({z}/{t})")

            except Exception:
                print(f"[{f.name}] : Erreur de Parsing")
