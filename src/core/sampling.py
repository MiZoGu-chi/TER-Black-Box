import numpy as np
import random
from abc import ABC, abstractmethod

from src.core.algorithm import local_search, best_improvement, get_black_box_fitness_fn

def generate_hamming_neighbors(reference_solution, distance, n_samples, rng):
    N = len(reference_solution)
    
    if distance == 0:
        return reference_solution[np.newaxis, :]
        
    neighbors = np.tile(reference_solution, (n_samples, 1))
    
    for i in range(n_samples):
        mutation_indices = rng.choice(N, size=min(distance, N), replace=False)
        neighbors[i, mutation_indices] = 1 - neighbors[i, mutation_indices]
        
    return neighbors

class EvaluationSampler(ABC):

    @abstractmethod
    def generate_samples(self, N, n_samples, seed, landscape=None):
        pass

    def _set_seed(self, seed): # faire un utils et mettre ça dedans plutot
        np.random.seed(seed)
        random.seed(seed)


class RandomSampler(EvaluationSampler):

    def generate_samples(self, N, n_samples, seed, landscape=None):
        self._set_seed(seed)
        return np.random.randint(2, size=(n_samples, N))


class StratifiedSampler(EvaluationSampler):
    def generate_samples(self, N, n_samples, seed, landscape=None):
        self._set_seed(seed)

        # * 10 arbitraire pour augmenter les chances d'avoir une bonne diversité de solutions
        pool_size = n_samples * 10
        candidate_pool = np.random.randint(2, size=(pool_size, N))
        fitness_values = get_black_box_fitness_fn(landscape)(candidate_pool)  # coûteux

        sorted_indices = np.argsort(fitness_values)
        n_per_stratum = n_samples // 3  # répartition en 3 groupes bon/moyen/mauvais
        stratum_capacity = len(sorted_indices) // 3

        low_fitness_set = sorted_indices[:stratum_capacity]
        medium_fitness_set = sorted_indices[stratum_capacity : 2 * stratum_capacity]
        high_fitness_set = sorted_indices[2 * stratum_capacity:]

        sampled_indices = np.concatenate([
            np.random.choice(low_fitness_set, n_per_stratum, replace=False),
            np.random.choice(medium_fitness_set, n_per_stratum, replace=False),
            np.random.choice(high_fitness_set, n_per_stratum, replace=False)
        ])

        return candidate_pool[sampled_indices]


class WhiteBoxStratifiedSampler(EvaluationSampler):
    def generate_samples(self, N, n_samples, seed, landscape=None):
        self._set_seed(seed)

        n_per_stratum = n_samples // 3
        objective_function = get_black_box_fitness_fn(landscape)

        # fonction inversée pour forcer la recherche locale vers des minima
        inverted_objective_function = lambda solutions: -objective_function(solutions)

        medium_fitness_set = np.random.randint(2, size=(n_per_stratum, N))

        high_fitness_set = np.array([
            local_search(np.random.randint(2, size=N), objective_function, best_improvement)
            for _ in range(n_per_stratum)
        ])

        low_fitness_set = np.array([
            local_search(np.random.randint(2, size=N), inverted_objective_function, best_improvement)
            for _ in range(n_per_stratum)
        ])

        return np.vstack([low_fitness_set, medium_fitness_set, high_fitness_set])

SAMPLERS = {
    'random': RandomSampler,
    'stratified': StratifiedSampler,
    'whitebox': WhiteBoxStratifiedSampler,
}