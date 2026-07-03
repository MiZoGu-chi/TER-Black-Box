from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.io_utils import OptimizationResult

# ajouter budget, strat de sélection ect ect
def build_ranking_dataframe(files):
    groups = defaultdict(list)

    for file in files:
        try:
            res = OptimizationResult(file)
            meta = res.meta
            key = (meta['N'], meta['K'], meta['algo'], meta['alpha'], meta['tabu_factor'], meta['drils_factor'], meta['budget'])
            last_score = res.fitness_history[-1][0]
            groups[key].append((last_score, meta.get('mean_local_optima')))
        except Exception as e:
            print(f"Erreur : {file.name} : {e}")

    rows = []
    for (N, K, algo, alpha, tabu_factor, drils_factor, budget), runs in groups.items():
        scores = [r[0] for r in runs]
        local_optima_vals = [r[1] for r in runs if r[1] is not None]
        mean_local_optima = np.mean(local_optima_vals) if local_optima_vals else None
        mean_score = np.mean(scores)
        amelioration = (mean_score - mean_local_optima) if mean_local_optima is not None else None
        rows.append({
            'N': N, 'K': K,
            'algo': algo, 'alpha': alpha, 'tabu_factor': tabu_factor, 'drils_factor': drils_factor, 'budget': budget,
            'score': mean_score, 'std': np.std(scores),
            'local_optima': mean_local_optima,
            'amelioration': amelioration,
            'n_runs': len(scores),
        })

    return pd.DataFrame(rows)
