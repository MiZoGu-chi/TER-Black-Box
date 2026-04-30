import ast
import numpy as np
import argparse
from pathlib import Path

def get_sparsity(filepath):
    with open(filepath, 'r') as f:

        lines = f.readlines()
        if not lines:
            return None
        
        coef_line = lines[-1].strip()

        coefs = np.array(ast.literal_eval(coef_line))
        zero_count = np.sum(coefs == 0.0)
        total = len(coefs)
        sparsity_pct = (zero_count / total) * 100
            
        return sparsity_pct, zero_count, total

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("opti_*.txt"))
    else:
        print(f"'{args.path}' n'est ni un fichier ni un dossier valide.")
        files = []

    for f in files:
        stats = get_sparsity(f)
        if stats:
            s, z, t = stats
            print(f"[{f.name}] : {s:.1f}% zeros ({z}/{t})")
        else: 
            print("erreur de parsing")
    
  