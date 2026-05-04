import ast
import numpy as np

class OptimizationResult:

    def __init__(self, filepath):

        self.filepath = filepath
        self.meta, self.coefs = self._parse_file()

    def _parse_file(self):

        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        header = lines[0].split()

        meta = {
            'N': int(header[0]),
            'D': int(header[2])
        }
        
        coefs = np.array(ast.literal_eval(lines[-1]))
        
        return meta, coefs

    @property
    def sparsity(self):
        
        zero_count = np.sum(self.coefs == 0.0)
        total = len(self.coefs)
        pct = (zero_count / total) * 100
        return pct, zero_count, total