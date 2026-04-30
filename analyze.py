import ast
import numpy as np
import argparse
import pprint

from pathlib import Path
from walsh import generate_masks

from collections import defaultdict

# factoriser la lecture de la dérnière ligne

def get_sparsity(filepath):
    with open(filepath, 'r') as f:

        lines = f.readlines()
        coef_line = lines[-1].strip()
        coefs = np.array(ast.literal_eval(coef_line))

        zero_count = np.sum(coefs == 0.0)
        total = len(coefs)
        sparsity_pct = (zero_count / total) * 100
            
        return sparsity_pct, zero_count, total

    return None

def get_dependency_graph(filepath, masks):
    with open(filepath, 'r') as f:

        lines = f.readlines()
        coef_line = lines[-1].strip()
        coefs = np.array(ast.literal_eval(coef_line))

        inferred_edges = []

        group_by_order = defaultdict(list)

        significant_coefs = np.where(coefs != 0.0)[0]

        for c in significant_coefs:
            mask = masks[c]
            features = np.where(mask == 1)[0].tolist()

            order = len(features)

            val = float(coefs[c]) # retourne un np.float64

            # on ignore le masque 0
            if order > 0: 
                interaction_group = tuple(sorted(features))
                group_by_order[order].append((interaction_group, round(val, 5)))

        return dict(group_by_order)            
 

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

    for file in files:
        stats = get_sparsity(file)
        if stats:
            s, z, t = stats
            print(f"[{file.name}] : {s:.1f}% zeros ({z}/{t})")

            # graphe d'interaction

            with open(file, 'r') as f:
                header = f.readline().strip().split()
                N = int(header[0])
                D = int(header[2])

            masks = generate_masks(N, D)

            # peut etre refaire sand donner file
            graph = get_dependency_graph(file, masks)

            pprint.pprint(graph, width=80, sort_dicts=True)

            #for ordre, interaction in sorted(graph.items()):
                #print(f"  -> Ordre {ordre} : {len(interactions)} interactions trouvées")
        else: 
            print("erreur de parsing")
    
  