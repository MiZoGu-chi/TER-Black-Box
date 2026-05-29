import ast
import numpy as np
from src.visualization import save_plot

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
            # nouveau format (5 sections) : meta | fitness | fidelity | coefs | best_sol
            elif len(sections) == 5 and sections[3].strip().startswith('['):
                meta_section, fitness_section, fidelity_section, coefs_section, best_sol_section = sections
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
                'expected_local_optima': float(header[7]) if len(header) > 7 else None
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


from src.visualization import save_df_as_image