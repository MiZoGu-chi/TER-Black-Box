import argparse
import pprint
import numpy as np
from pathlib import Path
from collections import defaultdict

from walsh import generate_masks
from io_utils import OptimizationResult

def get_dependency_graph(coefs, masks):

    group_by_order = defaultdict(list)

    significant_coefs = np.where(coefs != 0.0)[0]

    for c in significant_coefs:
        mask = masks[c]
        features = np.where(mask == 1)[0].tolist()
        order = len(features)

        if order > 0: # On ignore le masque 0
            interaction_group = tuple(sorted(features))
            val = float(coefs[c]) # coefs[c] retourne un np.float64
            group_by_order[order].append((interaction_group, round(val, 5)))

    return dict(group_by_order)        
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_dir():
        files = sorted(path.glob("opti_*.txt"))
    elif path.is_file():
        files = [path]
    else:
        print(f"{args.path} n'est ni un fichier ni un dossier valide.")
        files = []

    for file in files:
        try:
            res = OptimizationResult(file)
            
            s, z, t = res.sparsity_stats
            print(f"[{file.name}] : {s:.1f}% zeros ({z}/{t})")

            masks = generate_masks(res.meta['N'], res.meta['D'])

            graph = get_dependency_graph(res.coefs, masks)
            
            pprint.pprint(graph, width=80, sort_dicts=True)

        except Exception as e:
            print(f"Erreur lors de l'analyse de {file.name} : {e}")
    
  