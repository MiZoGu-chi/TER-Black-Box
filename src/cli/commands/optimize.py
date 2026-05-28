import os
import glob
from joblib import Parallel, delayed
from datetime import datetime
import numpy as np
import random
import multiprocessing
from pathlib import Path

from .base import Command 
from src.core.walsh import generate_masks
from src.core.nk import NK_Landscape
from src.core.surrogate_optimization import run_surrogate_optimization
from src.core.models import MODELS
from src.core.algorithm import ALGORITHMS, BASELINE_ALGORITHMS, START_STRATEGIES, local_search, best_improvement

from src.utils.io_utils import OptimizationResult

class OptimizeCommand(Command):

    def __init__(self, args):
        super().__init__(args)
        self.configs_N = args.N
        self.configs_K = args.K
        self.config_seeds = range(10)
        self.config_n_instances = range(10)
        self.algo = args.algo
        self.n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
        # self.njobs = 10

    @staticmethod
    def _task_wrapper(N, K, num, seed, args, output_dir, expected_local_optima):

        np.random.seed(seed)
        random.seed(seed)

        instance_name = f"instances/{N}/{K}/nk_{N}_{K}_{num}.txt"

        if not os.path.exists(instance_name):
            return

        landscape = NK_Landscape(instance_name)

        masks = generate_masks(N, args.D)
        model = MODELS[args.surrogate]['model']
        

        best_fit, best_sol, history, surrogate_model, surrogate_fidelity_history, final_scatter_data = run_surrogate_optimization(args.budget, N, args.initial_pop_size, landscape, masks, model, ALGORITHMS[args.algo], START_STRATEGIES[args.start])

        coefs = surrogate_model.coef_.tolist()

        filename = f"{output_dir}/opti_{N}_{K}_{num}_{seed}_{args.algo}_{args.surrogate}.txt"

        with open(filename, "w") as f:
            f.write(f"{N} {K} {args.D} {num} {seed} {args.algo} {args.surrogate} {expected_local_optima} \n")
            
            f.write("#\n")
            for sol, budg in history:
                f.write(f"{sol} {budg}\n")

            f.write("#\n")
            for pred, real, n_eval in surrogate_fidelity_history: # ajout coloration des points en fonction de n_evals
                f.write(f"{pred} {real} {n_eval}\n")

            f.write("#\n")
            for pred, real in final_scatter_data:
                f.write(f"{pred} {real}\n")

            f.write("#\n")                
            f.write(f"{coefs}\n")

    def execute(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/opti_output/run_{timestamp}") 
        output_dir.mkdir(parents=True, exist_ok=True)

        search_fn = BASELINE_ALGORITHMS[self.algo]
 
        baselines = {}
 
        for N in self.configs_N:
            for K in self.configs_K:
                for num in self.config_n_instances:
                    instance_name = f"instances/{N}/{K}/nk_{N}_{K}_{num}.txt"
                    if os.path.exists(instance_name):
                        landscape = NK_Landscape(instance_name)
                        baselines[(N, K, num)] = landscape.expected_local_optima(search_fn)
                        
        Parallel(n_jobs=self.n_jobs)(delayed(self._task_wrapper)(N, K, num, seed, self.args, output_dir, baselines[(N, K, num)]) 
            for N in self.configs_N 
            for K in self.configs_K 
            for num in self.config_n_instances
            for seed in self.config_seeds
        )


        result_files = sorted(output_dir.glob("opti_*.txt"))

        for f in result_files:
            try:
                res = OptimizationResult(f)
                s, z, t = res.sparsity
                print(f"[{f.name}] : {s:.1f}% zeros ({z}/{t})")

            except Exception:
                print(f"[{f.name}] : Erreur de Parsing")