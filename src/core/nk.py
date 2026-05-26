from collections import defaultdict
from itertools import combinations
from pathlib import Path

from src.core.algorithm import get_black_box_fitness_fn

import numpy as np
import json


class NK_Landscape:

    def __init__(self, file, max_nb_turn=-1):

        f = open(file, "r")
        lignes = f.readlines()

        head = lignes[0].split()

        self.N = int(head[0])
        self.K = int(head[1])

        if (len(head) > 2):
            self.D = int(head[2])
        else:
            self.D = 2

        self.links = []
        for n in range(self.N):
            self.links.append([])
            for k in range(self.K + 1):
                self.links[n].append(int(lignes[1 + n * (self.K + 1) + k]))

            self.links[n].append([])
            for k in range(self.D ** (self.K + 1)):
                self.links[n][-1].append(float(lignes[1 + self.N * (self.K + 1) + n * (self.D ** (self.K + 1)) + k]))

        f.close()

        # un seul fichier cache par dossier N K
        self._cache_path = Path(file).parent / "expected_local_optima.json"

        # on extrait le numéro de l'isinstance
        self._cache_instance_key = Path(file).stem.split('_')[-1]

    def eval(self, solution):

        fitness = 0

        for link in self.links:
            locus = []
            for i in link[:-1]:
                locus.append(solution[i])
            locus_index = 0
            for i in range(self.K + 1):
                locus_index += (self.D ** (self.K - i)) * locus[i]

            fitness += link[-1][int(locus_index)]

        return fitness / self.N

    def get_real_interaction(self):

        interactions = defaultdict(set)

        for link in self.links:

            # On enlève le dernier élément (la table de fitness)
            # on sort pour ne pas avoir des paires (x, y) et (y, x)
            neighborhood = sorted(link[:-1])

            for order in range(1, self.K + 2):

                # combinations retourne déjà des tuples, pas besoin de tuple()
                for interaction_group in combinations(neighborhood, order):
                    interactions[order].add(interaction_group)

        return {order: sorted(groups) for order, groups in interactions.items()}

    def expected_local_optima(self, search_fn, cache_key, n_runs=100, seed=0):

        config_key = f"{cache_key}_S{seed}_R{n_runs}"

        cache = self._load_cache()

        if self._cache_instance_key in cache and config_key in cache[self._cache_instance_key]:
            self._expected_local_optima = cache[self._cache_instance_key][config_key]
            return self._expected_local_optima

        local_seed = np.random.default_rng(seed)
        fit_fn = get_black_box_fitness_fn(self)
        fitness = []

        for _ in range(n_runs):
            start_sol = local_seed.integers(2, size=self.N)
            local_optimum = search_fn(start_sol, fit_fn)
            fitness.append(self.eval(local_optimum))

        result = float(np.mean(fitness))

        # à enlever potentielement
        cache.setdefault(self._cache_instance_key, {})[config_key] = result
        self._save_cache(cache)

        self._expected_local_optima = result
        return result

    def _load_cache(self):
        if self._cache_path.exists():
            with open(self._cache_path) as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache):
        with open(self._cache_path, 'w') as f:
            json.dump(cache, f, indent=2)