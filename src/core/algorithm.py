# algorithm.py
import numpy as np

from src.core.local_search import local_search, tabu_search, best_improvement, first_improvement
from src.core.cpp.surrogate_search import HillClimber, TabuSearch, DRILS


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
    if name == 'cpp_hc':
        return lambda masks, N: search(HillClimber(masks_to_indices(masks), N, HillClimber.Strategy.BEST_IMPROVEMENT))
    if name == 'cpp_hc_fi':
        return lambda masks, N: search(HillClimber(masks_to_indices(masks), N, HillClimber.Strategy.FIRST_IMPROVEMENT))
    if name == 'cpp_tabu':
        return lambda masks, N: search(TabuSearch(masks_to_indices(masks), N, N * tabu_factor))
    if name == 'cpp_drils_hc_bi':
        return lambda masks, N: search(DRILS(masks_to_indices(masks), N, DRILS.LocalSearch.HC_BEST_IMPROVEMENT,
            alpha=perturbation_strength, max_iter=N * drils_factor, tabu_max_iter=0))
    if name == 'cpp_drils_hc_fi':
        return lambda masks, N: search(DRILS(masks_to_indices(masks), N, DRILS.LocalSearch.HC_FIRST_IMPROVEMENT,
            alpha=perturbation_strength, max_iter=N * drils_factor, tabu_max_iter=0))
    if name == 'cpp_drils_tabu':
        return lambda masks, N: search(DRILS(masks_to_indices(masks), N, DRILS.LocalSearch.TABU,
            alpha=perturbation_strength, max_iter=N * drils_factor, tabu_max_iter=N * tabu_factor))
    raise ValueError(f"Algorithme inconnu : {name}")


def make_algorithm(name, masks, N, **hyperparams):
    return get_algorithm_factory(name, **hyperparams)(masks, N)


ALL_ALGORITHM_NAMES = ['cpp_hc', 'cpp_hc_fi', 'cpp_tabu', 'cpp_drils_hc_bi', 'cpp_drils_hc_fi', 'cpp_drils_tabu']


_BASELINES = {
    'fi':   fi_baseline,
    'bi':   bi_baseline,
    'tabu': tabu_search,
}

ALL_BASELINE_NAMES = list(_BASELINES.keys())

def make_baseline(name):
    if name not in _BASELINES:
        raise ValueError(f"Baseline inconnue : {name}")
    return _BASELINES[name]