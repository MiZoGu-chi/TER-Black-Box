import ast
import numpy as np

class OptimizationResult:

    def __init__(self, filepath):

        self.filepath = filepath
        self.meta, self.coefs, self.fitness_history, self.fidelity_history = self._parse_file()

    def _parse_file(self):

        with open(self.filepath, 'r') as f:
            lines = f.read()
        
        meta_section, fitness_section, fidelity_section, coefs_section = lines.split("\n#\n")

        header = meta_section.strip().split()

        meta = {
            'N': int(header[0]),
            'K': int(header[1]),
            'D': int(header[2]),
            'num' : int(header[3]),
            'seed': int(header[4]),
            'algo': header[5],      
            'surrogate': header[6]  
        }
        
        fitness_history = [(float(line.split()[0]), int(line.split()[1])) for line in fitness_section.strip().split('\n')]
        fidelity_history = [(float(line.split()[0]), float(line.split()[1]), int(line.split()[2])) for line in fidelity_section.strip().split('\n')]    

        coefs = np.array(ast.literal_eval(coefs_section))
        
        return meta, coefs, fitness_history, fidelity_history

    @property
    def sparsity(self):
        
        zero_count = np.sum(self.coefs == 0.0)
        total = len(self.coefs)
        pct = (zero_count / total) * 100
        return pct, zero_count, total