import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from src.core.data_sampler import transform
from src.core.sampling import generate_hamming_neighbors
from src.core.models import MODELS

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


def create_metrics_dataframe(confusion_matrix_by_order, N, K):
    tp, fp, fn, precision, recall, f1 = compute_confusion_metrics(confusion_matrix_by_order)

    df = pd.DataFrame({
        'N': [N] * len(tp),
        'K': [K] * len(tp),
        'D': range(1, len(tp) + 1),
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })

    for col in ['Precision', 'Recall', 'F1']:
        df[col] = df[col].map("{:.2%}".format)

    return df


def compute_fidelity_mae(fidelity_history):
    preds = [p[0] for p in fidelity_history]
    reals = [p[1] for p in fidelity_history]
    return mean_absolute_error(reals, preds)


def load_fitted_model(surrogate_name, coefs):
    model = clone(MODELS[surrogate_name]['model'])
    model.coef_ = np.array(coefs)
    model.intercept_ = 0.0
    return model


def distance_grid(N, n_points):
    return np.unique(np.round(np.linspace(0, N // 2, n_points)).astype(int))


def evaluate_global_fidelity(model, masks, fit_fn, test_solutions):
    real_fitness = fit_fn(test_solutions)
    pred_fitness = model.predict(transform(masks, test_solutions))
    return np.column_stack([pred_fitness, real_fitness])


def evaluate_covariate_shift(model, masks, fit_fn, best_sol, run_seed, samples, n_points=9):
    covariate_data = defaultdict(list)
    covariate_scatter = defaultdict(lambda: {'real': [], 'pred': []})

    rng = np.random.default_rng(run_seed)

    for dist in distance_grid(len(best_sol), n_points):
        sols = generate_hamming_neighbors(best_sol, int(dist), samples // 10, rng)
        real_f = fit_fn(sols)
        pred_f = model.predict(transform(masks, sols))

        covariate_data[int(dist)].extend(np.abs(pred_f - real_f).tolist())
        covariate_scatter[int(dist)]['real'].extend(real_f.tolist())
        covariate_scatter[int(dist)]['pred'].extend(pred_f.tolist())

    return covariate_data, covariate_scatter

