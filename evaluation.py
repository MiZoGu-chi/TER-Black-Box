import numpy as np
import pandas as pd
from collections import defaultdict

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

    confusion_matrix_by_K = np.zeros((D, 3), dtype=int)

    for order in range(1, D + 1):
        surrogate_interactions = set(item[0] for item in surrogate_graph.get(order, []))
        real_interactions = set(real_graph.get(order,[]))

        tp = len(surrogate_interactions.intersection(real_interactions))
        fp = len(surrogate_interactions - real_interactions)
        fn = len(real_interactions - surrogate_interactions)  

        confusion_matrix_by_K[order - 1] = [tp, fp, fn]     
    
    return confusion_matrix_by_K

def compute_confusion_metrics(confusion_matrix_by_K):

    tp = confusion_matrix_by_K[:, 0]
    fp = confusion_matrix_by_K[:, 1]
    fn = confusion_matrix_by_K[:, 2]   

    # Gestion de la division par 0
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / np.maximum(tp + fn, 1e-9)
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-9)

    return tp, fp, fn, precision, recall, f1

