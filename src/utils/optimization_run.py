from dataclasses import dataclass
from pathlib import Path
import os
import random

import numpy as np

from src.core.nk import NK_Landscape
from src.core.surrogate_optimization import run_surrogate_optimization
from src.utils.io_utils import write_optimization_result, load_landscape_and_masks, instance_path


@dataclass
class RunConfig:
    D: int
    algo: str
    surrogate_name: str
    model: object
    make_algorithm_fn: object
    baseline_factory: object
    start_fn: object
    budget: int
    initial_pop_size: int
    duplicate_strategy: str
    tabu_factor: int
    drils_factor: int


def load_baselines(N_list, K_list, num_list, baseline_factory):
    baselines = {}

    for N in N_list:
        for K in K_list:
            for num in num_list:
                instance_name = instance_path(N, K, num)
                if os.path.exists(instance_name):
                    landscape = NK_Landscape(instance_name)
                    baselines[(N, K, num)] = landscape.mean_local_optima(baseline_factory(landscape))

    return baselines


def output_dir(config, N, K, timestamp):
    path = Path("results/opti_output") / str(N) / str(config.D) / str(K) / config.surrogate_name

    if 'alpha' in config.model.get_params():
        path = path / f"{config.model.alpha:g}"

    path = path / config.algo
    if 'tabu' in config.algo:
        path = path / str(config.tabu_factor)
    elif 'drils' in config.algo:
        path = path / str(config.drils_factor)

    return path / f"run_{timestamp}"


def result_file_path(config, N, K, num, seed, timestamp):
    return output_dir(config, N, K, timestamp) / f"opti_{N}_{K}_{num}_{seed}_{config.algo}_{config.surrogate_name}.txt"


def run_and_save(config, N, K, num, seed, timestamp, mean_local_optima):

    np.random.seed(seed)
    random.seed(seed)

    landscape, masks = load_landscape_and_masks(N, K, config.D, num)

    alg = config.make_algorithm_fn(masks, N)

    best_fit, best_sol, history, surrogate_model, fidelity_history, alpha_history = run_surrogate_optimization(
        config.budget, N, config.initial_pop_size, landscape, masks, config.model, alg, config.start_fn, config.duplicate_strategy)

    algo_factor = config.drils_factor if 'drils' in config.algo else config.tabu_factor
    header_alpha = config.model.alpha if config.surrogate_name == 'lasso_lars' else None

    write_optimization_result(
        result_file_path(config, N, K, num, seed, timestamp),
        N, K, config.D, num, seed, config.algo, config.surrogate_name,
        mean_local_optima, header_alpha, algo_factor, config.duplicate_strategy,
        history, fidelity_history, best_sol, surrogate_model.coef_.tolist(),
        alpha_history, config.surrogate_name == 'lasso_lars_ic',
    )
