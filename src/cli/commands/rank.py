from collections import defaultdict

import numpy as np
import pandas as pd

from .base import Command
from src.utils.io_utils import OptimizationResult
from src.visualization.ranking import plot_score_vs_alpha, plot_heatmap_alpha_tabu

class RankCommand(Command):

    def execute(self):
        files = self._get_files()

        groups = defaultdict(list)
        for file in files:
            try:
                res = OptimizationResult(file)
                meta = res.meta
                key = (meta['N'], meta['K'], meta['alpha'], meta['tabu_factor'])
                last_score = res.fitness_history[-1][0]
                groups[key].append((last_score, meta.get('mean_local_optima')))
            except Exception as e:
                print(f"Erreur : {file.name} : {e}")

        rows = []
        for (N, K, alpha, tabu_factor), runs in groups.items():
            scores = [r[0] for r in runs]
            local_optima_vals = [r[1] for r in runs if r[1] is not None]
            mean_local_optima = np.mean(local_optima_vals) if local_optima_vals else None
            mean_score = np.mean(scores)
            amelioration = (mean_score - mean_local_optima) if mean_local_optima is not None else None
            rows.append({
                'N': N, 'K': K,
                'alpha': alpha, 'tabu_factor': tabu_factor,
                'score': mean_score, 'std': np.std(scores),
                'local_optima': mean_local_optima,
                'amelioration': amelioration,
                'n_runs': len(scores),
            })

        df = pd.DataFrame(rows)

        for (N, K), group in df.groupby(['N', 'K']):
            group = group.sort_values('score', ascending=False).reset_index(drop=True)
            group.index += 1

            n_runs = group['n_runs'].iloc[0]
            print(f"\n{'='*60}")
            print(f"N={N}, K={K}  ({n_runs} runs/config)")
            print(f"{'='*60}")

            display = group[['alpha', 'tabu_factor', 'score', 'std', 'amelioration']].copy()
            display['amelioration'] = display['amelioration'].map(lambda x: f"{x:+.4f}" if x is not None else "N/A")
            print(display.to_string(float_format=lambda x: f"{x:.4f}"))

        plot_score_vs_alpha(df)
        plot_heatmap_alpha_tabu(df)
