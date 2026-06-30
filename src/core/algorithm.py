# algorithm.py
import numpy as np

from src.core.local_search import local_search, tabu_search, best_improvement, first_improvement
from src.core.cpp.surrogate_search import HillClimber, TabuSearch, DRILS
from src.core.drils import drils_nk

ALGORITHM_NAMES = [
    'hc_fi', 
    'hc_bi', 
    'tabu', 
    'drils_hc_bi', 
    'drils_hc_fi', 
    'drils_tabu'
]

def fi_baseline(sol, fit_fn):
    return local_search(sol, fit_fn, first_improvement)

def bi_baseline(sol, fit_fn):
    return local_search(sol, fit_fn, best_improvement)

def masks_to_indices(masks): # [0,1,0,0,1,1] ->  [1,5,6]
    return [np.where(m)[0].tolist() for m in masks]

def search(searcher):
    def search(sol, model):
        return searcher.search(
            np.asarray(sol, dtype=np.int32),
            np.asarray(model.coef_, dtype=np.float64),
            np.random.randint(0, 2**31),
        )
    return search


def get_algorithm_factory(name, tabu_factor=None, perturbation_strength=None, drils_factor=None):
    if name == 'hc':
        return lambda masks, N: search(HillClimber(masks_to_indices(masks), N, HillClimber.Strategy.BEST_IMPROVEMENT))
    if name == 'hc_fi':
        return lambda masks, N: search(HillClimber(masks_to_indices(masks), N, HillClimber.Strategy.FIRST_IMPROVEMENT))
    if name == 'tabu':
        return lambda masks, N: search(TabuSearch(masks_to_indices(masks), N, N * tabu_factor))
    if name == 'drils_hc_bi':
        return lambda masks, N: search(DRILS(masks_to_indices(masks), N, DRILS.LocalSearch.HC_BEST_IMPROVEMENT,
            alpha=perturbation_strength, max_iter=N * drils_factor, tabu_max_iter=0))
    if name == 'drils_hc_fi':
        return lambda masks, N: search(DRILS(masks_to_indices(masks), N, DRILS.LocalSearch.HC_FIRST_IMPROVEMENT,
            alpha=perturbation_strength, max_iter=N * drils_factor, tabu_max_iter=0))
    if name == 'drils_tabu':
        return lambda masks, N: search(DRILS(masks_to_indices(masks), N, DRILS.LocalSearch.TABU,
            alpha=perturbation_strength, max_iter=N * drils_factor, tabu_max_iter=N * tabu_factor))
    raise ValueError(f"Algorithme inconnu : {name}")


def make_baseline(name, alpha=None, drils_factor=None):
    match name:
        case 'hc_fi':
            return lambda landscape: bi_baseline
        case 'hc_bi':
            return lambda landscape: fi_baseline
        case 'tabu':
            return lambda landscape: tabu_search
        case 'drils_hc_bi' | 'drils_hc_fi' | 'drils_tabu':
            inner_ls = {'drils_hc_bi': bi_baseline, 'drils_hc_fi': fi_baseline, 'drils_tabu': tabu_search}[name]
            def factory(landscape):
                vig = landscape.get_variable_interaction_graph()
                max_iter = landscape.N * drils_factor
                return lambda start_sol, fit_fn: drils_nk(landscape, vig, inner_ls, alpha, max_iter, start_sol)
            return factory
        case _:
            raise ValueError(f"Baseline inconnue : {name}")