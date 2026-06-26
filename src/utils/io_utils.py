import ast
import os
import numpy as np
from src.visualization import save_plot
from src.core.nk import NK_Landscape
from src.core.walsh import generate_masks

class OptimizationResult:

    def __init__(self, filepath):

        self.filepath = filepath
        self.meta, self.coefs, self.fitness_history, self.fidelity_history, self.final_scatter, self.best_sol = self._parse_file()

    def _parse_file(self):
            with open(self.filepath, 'r') as f:
                raw = f.read()

            sections = raw.split("\n#\n")

            best_sol_section = None

            # ancien format (5 sections) : meta | fitness | fidelity | scatter | coefs
            if len(sections) == 5 and not sections[3].strip().startswith('['):
                meta_section, fitness_section, fidelity_section, _, coefs_section = sections
            # nouveau format (5 sections) : meta | fitness | fidelity | best_sol | coefs
            elif len(sections) == 5 and sections[3].strip().startswith('['):
                meta_section, fitness_section, fidelity_section, best_sol_section, coefs_section = sections
            elif len(sections) == 4:
                meta_section, fitness_section, fidelity_section, coefs_section = sections
            else:
                raise ValueError(f"Format de fichier inconnu : {len(sections)} sections")

            # 1. Méta
            header = meta_section.strip().split()
            meta = {
                'N': int(header[0]),
                'K': int(header[1]),
                'D': int(header[2]),
                'num' : int(header[3]),
                'seed': int(header[4]),
                'algo': header[5],
                'surrogate': header[6],
                'mean_local_optima': float(header[7]) if len(header) > 7 else None,
                'alpha': float(header[8]) if len(header) > 8 else None,
                'tabu_factor': int(header[9]) if len(header) > 9 else None,
            }

            # 2. Fitness
            fitness_history = [(float(line.split()[0]), int(line.split()[1])) for line in fitness_section.strip().split('\n')]

            # 3. Fidelity
            fidelity_history = [
                (float(line.split()[0]), float(line.split()[1]), int(line.split()[2]))
                for line in fidelity_section.strip().split('\n') if line.strip()
            ]

            # 4. Coefs
            coefs = np.array(ast.literal_eval(coefs_section.strip()))
            final_scatter = np.array([])

            # 5. best_sol (optionnel)
            best_sol = np.array(ast.literal_eval(best_sol_section.strip()), dtype=int) if best_sol_section else None

            return meta, coefs, fitness_history, fidelity_history, final_scatter, best_sol

    @property
    def sparsity(self):

        zero_count = np.sum(self.coefs == 0.0)
        total = len(self.coefs)
        pct = (zero_count / total) * 100

        return pct, zero_count, total


def write_optimization_result(path, N, K, D, num, seed, algo, surrogate_name, mean_local_optima,
                               header_alpha, algo_factor, duplicate_strategy,
                               history, fidelity_history, best_sol, coefs,
                               alpha_history, write_alpha_history):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"{N} {K} {D} {num} {seed} {algo} {surrogate_name} {mean_local_optima} {header_alpha} {algo_factor} {duplicate_strategy}\n")

        f.write("#\n")
        for sol, budg in history:
            f.write(f"{sol} {budg}\n")

        f.write("#\n")
        for pred, real, n_eval in fidelity_history:
            f.write(f"{pred} {real} {n_eval}\n")

        f.write("#\n")
        f.write(f"{best_sol.tolist()}\n")

        f.write("#\n")
        f.write(f"{coefs}\n")

        if write_alpha_history:
            f.write("#\n")
            for n_eval, alpha_ in alpha_history:
                f.write(f"{n_eval} {alpha_}\n")


def instance_path(N, K, num):
    return f"instances/{N}/{K}/nk_{N}_{K}_{num}.txt"


def load_landscape_and_masks(N, K, D, num):
    instance_name = instance_path(N, K, num)

    if not os.path.exists(instance_name):
        raise FileNotFoundError(f"L'instance {instance_name} est introuvable.")

    landscape = NK_Landscape(instance_name)
    masks = generate_masks(N, D)

    return landscape, masks


def build_plot_tag(N, K, surrogate, algo, alpha=None, tabu_factor=None):
    subdir = f"{N}/{K}/{surrogate}"
    if alpha is not None:
        subdir += f"/{alpha:g}"
    subdir += f"/{algo}"
    if tabu_factor is not None and 'tabu' in algo:
        subdir += f"/{tabu_factor}"
    return subdir, subdir.replace('/', '_')


from src.visualization import save_df_as_image