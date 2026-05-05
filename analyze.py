import argparse
import pprint
import numpy as np
from pathlib import Path
from collections import defaultdict

from walsh import generate_masks
from io_utils import OptimizationResult
from nk import NK_Landscape

def get_surrogate_interaction(coefs, masks):

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
            
            s, z, t = res.sparsity
            print(f"[{file.name}] : {s:.1f}% zeros ({z}/{t})")

            N = res.meta['N']
            K = res.meta['K']
            D = res.meta['K']
            seed = res.meta['seed']

            masks = generate_masks(N, D)

            # Format : { ordre1 : [((tuple1), coef1), ..., ((tupleN), coefN)], ordreN: ... }
            # Interactions + poids
            surrogate_graph = get_surrogate_interaction(res.coefs, masks)

            pprint.pprint(surrogate_graph, width=80, sort_dicts=True)

            # On est obligé de recréer l'instance NK
            instance_name = f"instances/{N}/{K}/nk_{N}_{K}_0.txt"
            landscape = NK_Landscape(instance_name)

            # Format : { ordre1 : [(tuple1), ..., (tupleN)], ordreN: ... }
            # Uniquement les intéractions 
            real_graph = landscape.get_real_interaction()

            for order in (1, D + 1):
                # on extrait uniquement les tuples (item[0]) pour un ordre donné et transforme en set
                surrogate_interactions = set(item[0] for item in surrogate_graph.get(order, []))
                real_interactions = set(real_graph.get(order, []))
                
                tp = surrogate_interactions.intersection(real_interactions)
                fp = surrogate_interactions - real_interactions
                fn = real_interactions - surrogate_interactions
                
                print(f"Ordre {order}:")
                print(f"Vrais Positifs : {len(tp)}")
                print(f"Faux Positifs : {len(fp)}")
                print(f"Faux Négatifs : {len(fn)}")

        except Exception as e:
            print(f"Erreur lors de l'analyse de {file.name} : {e}")
    
  