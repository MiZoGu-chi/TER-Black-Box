import numpy as np


def get_random_start(_best_sol, N):
    return np.random.randint(2, size=N)

def get_best_start(best_sol, N):
    return best_sol.copy()

def get_e_greedy_start(best_sol, N, **kwargs):
    epsilon = kwargs.get('epsilon', 0.3)
    if np.random.random() < epsilon:
        return np.random.randint(2, size=N)
    return best_sol.copy()


START_STRATEGIES = {
    'rd':      get_random_start,
    'best':    get_best_start,
    'egreedy': get_e_greedy_start,
}