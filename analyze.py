import argparse
import pprint
import numpy as np
import pandas as pd

from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_absolute_error

from walsh import generate_masks
from io_utils import OptimizationResult
from nk import NK_Landscape
from plot import plot_optimization, plot_aggregated_fidelity, plot_aggregated_fitness_distributions, plot_aggregated_convergence, save_df_as_image


# à transformer en commande 

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
 
def compare_interaction(surrogate_graph, real_graph, D):

    interactions = np.zeros((D, 3), dtype=int)

    for order in range(1, D + 1):
        surrogate_interactions = set(item[0] for item in surrogate_graph.get(order, []))
        real_interactions = set(real_graph.get(order,[]))

        tp = len(surrogate_interactions.intersection(real_interactions))
        fp = len(surrogate_interactions - real_interactions)
        fn = len(real_interactions - surrogate_interactions)  

        interactions[order - 1] = [tp, fp, fn]     
    
    return interactions

def analyse_optimization_file(file):

    res = OptimizationResult(file)
    
    s, z, t = res.sparsity

    N = res.meta['N']
    K = res.meta['K']
    D = res.meta['D']
    num = res.meta['num']
    seed = res.meta['seed']

    masks = generate_masks(N, D)

    # Format : { ordre1 : [((tuple1), coef1), ..., ((tupleN), coefN)], ordreN: ... }
    # Interactions + poids
    surrogate_graph = get_surrogate_interaction(res.coefs, masks)

    # On est obligé de recréer l'instance NK
    instance_name = f"instances/{N}/{K}/nk_{N}_{K}_{num}.txt"
    landscape = NK_Landscape(instance_name)

    # Format : { ordre1 : [(tuple1), ..., (tupleN)], ordreN: ... }
    # Uniquement les intéractions 
    real_graph = landscape.get_real_interaction()

    metrics = compare_interaction(surrogate_graph, real_graph, D)

    return {
        'filename' : file.name, 
        'K': K,
        'surrogate_graph' : surrogate_graph,
        'real_graph' : real_graph,
        'sparsity' : res.sparsity,
        'metrics' : metrics,
        'fidelity_history': res.fidelity_history, 
        'fitness_history': res.fitness_history        
    }

def create_metrics_dataframe(matrix):
    df = pd.DataFrame(matrix, columns=['TP', 'FP', 'FN'])
    df.index = range(1, len(df) + 1)
    df.index.name = 'K'  

    # gestion de la division par 0
    df['Precision'] = df['TP'] / np.maximum(df['TP'] + df['FP'], 1e-9)
    df['Recall'] = df['TP'] / np.maximum(df['TP'] + df['FN'], 1e-9)
    df['F1'] = 2 * (df['Precision'] * df['Recall']) / np.maximum(df['Precision'] + df['Recall'], 1e-9)

    for col in ['Precision', 'Recall', 'F1']:
            df[col] = df[col].map("{:.2%}".format) 
                  
    return df

def display_res(res):

    s, z, t = res['sparsity']

    # print("Graphe d'interaction du modèle surrogate")
    # pprint.pprint(res['surrogate_graph'], width=80, sort_dicts=True)

    print(f"\nMétriques - [{res['filename']}] : {s:.1f}% zeros ({z}/{t})")

    print(create_metrics_dataframe(res['metrics']))


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

    agreged_matrix = None

    maes = []

    max_fitness_by_K = defaultdict(list)
    fitness_trajectories_by_K = defaultdict(list)
    fidelity_samples_by_K = defaultdict(list)

    for file in files:
        try:
            result_data = analyse_optimization_file(file)

            K = result_data['K']
            metrics = result_data['metrics']

            if agreged_matrix is None: agreged_matrix = np.zeros_like(metrics)

            agreged_matrix += metrics

            display_res(result_data)

            #plot_optimization(file)        

            if result_data['fidelity_history']:
                fidelity_samples_by_K[K].extend(result_data['fidelity_history'])

                preds = [pred for pred, _, _ in result_data['fidelity_history']]
                reals = [real for _, real, _ in result_data['fidelity_history']]

                mae = mean_absolute_error(reals, preds)
                maes.append(mae)

            if result_data['fitness_history']:
                fitness_trajectories_by_K[K].append(result_data['fitness_history'])

                last_score = result_data['fitness_history'][-1][0]
                max_fitness_by_K[K].append(last_score)
            
        except Exception as e:
            print(f"Erreur lors de l'analyse de {file.name} : {e}")


    df_metrics = create_metrics_dataframe(agreged_matrix)
    #print(create_metrics_dataframe(agreged_matrix))

    print(f'mae surrogate : {np.mean(maes)}')

    for K, scores in sorted(max_fitness_by_K.items()):
        print(f'Score agrégé sur {len(scores)} runs pour K={K}: {np.mean(scores):.4f}')
        print(f'Ecart-type sur {len(scores)} runs pour K={K}: {np.std(scores):.4f}')
    
    plot_aggregated_fitness_distributions(max_fitness_by_K)
    plot_aggregated_convergence(fitness_trajectories_by_K)
    plot_aggregated_fidelity(fidelity_samples_by_K)
    save_df_as_image(df_metrics)
  