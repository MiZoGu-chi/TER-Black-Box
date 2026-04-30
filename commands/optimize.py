import os
import glob
from joblib import Parallel, delayed
from datetime import datetime
import numpy as np
import random
import multiprocessing
from pathlib import Path

from .base import Command 
from analyze import get_sparsity
from walsh import generate_masks
from nk import NK_Landscape
from optimization import run_surrogate_optimization
from models import MODELS
from algorithm import ALGORITHMS, START_STRATEGIES

class OptimizeCommand(Command):
    # à découpler de args plus tard
    def __init__(self, args):
        super().__init__(args)
        self.configs_N = [64]
        self.configs_K = [2]
        self.config_seeds = range(10)
        self.n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    @staticmethod
    def _task_wrapper(N, K, seed, args, output_dir):

        np.random.seed(seed)
        random.seed(seed)

        # à modifier (ne pas demander N et K ds le parser)
        args.N = N
        args.K = K

        instance_name = f"instances/{N}/{K}/nk_{N}_{K}_{args.num}.txt"

        if not os.path.exists(instance_name):
            return

        landscape = NK_Landscape(instance_name)
        masks = generate_masks(N, args.D)
        model = MODELS[args.surrogate]['model']
        

        best_fit, best_sol, history, surrogate_model = run_surrogate_optimization(args, landscape, masks, model, ALGORITHMS[args.algo], START_STRATEGIES[args.start])

        coefs = surrogate_model.coef_.tolist()

        filename = f"{output_dir}/opti_{N}_{K}_{seed}_{args.algo}_{args.surrogate}.txt"

        with open(filename, "w") as f:
            f.write(f"{N} {K} {args.D} {seed} {args.algo} {args.surrogate}\n")
            
            for sol, budg in history:
                f.write(f"{sol} {budg}\n")
                
            f.write(f"{coefs}\n")

    def execute(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"opti_output/run_{timestamp}") 
        output_dir.mkdir(parents=True, exist_ok=True)
 
        Parallel(n_jobs=self.n_jobs)(delayed(self._task_wrapper)(N, K, seed, self.args, output_dir) 
            for N in self.configs_N 
            for K in self.configs_K 
            for seed in self.config_seeds
        )

        result_files = sorted(output_dir.glob("opti_*.txt"))

        for f in result_files:
            stats = get_sparsity(f)
            if stats:
                s, z, t = stats
                print(f"[{f.name}] : {s:.1f}% zeros ({z}/{t})")
            else:
                print(f"[{f.name}] : Erreur de Parsing")