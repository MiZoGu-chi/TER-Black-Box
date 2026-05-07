import numpy as np
import random
from data_sampler import transform

def get_neighborhood(solution):
    N = len(solution)

    neighborhood = np.tile(solution, (N, 1)) # matrice N*N ou chaque ligne est une copie de la solution

    neighborhood[np.arange(N), np.arange(N)] = 1 - solution 
    # (N = 4) neighborhood[[0,1,2,3], [0,1,2,3]] -> retourne une liste des points [neigh[0,0], neigh[1,1], neigh[2,2], neigh[3,3]] ; il suffit alors de soustraire l'inverse de la solution (1 - solution)

    return neighborhood

def tabu_search(current_best, fitness_fn):
    N = len(current_best)
    max_iter = N * 2
    current_solution = current_best.copy()

    params1 = int(N // 10) 
    params2 = 10

    best_solution = current_solution.copy()
    current_fitness = fitness_fn([current_solution])[0]
    best_fitness = current_fitness

    tabu_tenure = np.zeros((N))

    # pas de budget lorsqu'on utilise le modéle surrogate
    for iter_ in range(max_iter):

        # Génération et évalutaion de tous les voisins d'un coup
        neighborhood = get_neighborhood(current_solution)
        neighborhood_fitness = fitness_fn(neighborhood)

        best_delta = -float("inf")
        best_x = -1
        trouve = 1

        for x in range(N):
            new_fitness = neighborhood_fitness[x]

            delta = new_fitness - current_fitness

            if (tabu_tenure[x] <= iter_) or (new_fitness > best_fitness):

                if (delta > best_delta):
                    best_x = x 
                    best_delta = delta 
                    trouve = 1 

                elif (delta == best_delta):
                    trouve += 1 

                    if (random.randint(0, trouve) == 0): 
                        best_x = x

            # plus besoin de vérifier le budget

        current_fitness += best_delta
        current_solution[best_x] = 1 - current_solution[best_x]

        if (current_fitness > best_fitness): 
            best_fitness = current_fitness
            best_solution = np.copy(current_solution)    

        tabu_tenure[best_x] = int(params1) + random.randint(0, int(params2)) + iter_

    return best_solution

def best_improvement(neighborhood_fitness, x_star_fitness):
    best_idx = np.argmax(neighborhood_fitness)

    if neighborhood_fitness[best_idx] > x_star_fitness:
        return best_idx
    return None

def first_improvement(neighborhood_fitness, x_star_fitness):
    indices = np.random.permutation(len(neighborhood_fitness))

    for i in indices:
        if neighborhood_fitness[i] > x_star_fitness:
            return i
    return None

def get_surrogate_fitness_fn(surrogate_model, masks): 

    def fitness_fn(solutions):
        X_transformed = transform(masks, solutions)

        fitness = surrogate_model.predict(X_transformed)

        return fitness

    return fitness_fn

def get_black_box_fitness_fn(landscape):

    def fitness_fn(solutions):
        fitness = [landscape.eval(sol) for sol in solutions]

        return np.array(fitness)

    return fitness_fn

def local_search(current_best, fitness_fn, strategy_fn):
    x_star = current_best.copy()
    x_star_fitness = fitness_fn([x_star])[0]

    is_local_optimum = False

    while not is_local_optimum:

        neighborhood = get_neighborhood(x_star)
 
        neighborhood_fitness = fitness_fn(neighborhood) # pas idéal pour FI de tout calculer d'un coup ?
        
        best_idx = strategy_fn(neighborhood_fitness, x_star_fitness)

        if best_idx is not None:
            x_star = neighborhood[best_idx]
            x_star_fitness = neighborhood_fitness[best_idx]
        else:
            is_local_optimum = True

    return x_star

def get_random_start(_best_sol, N): 
    return np.random.randint(2, size=N)

def get_best_start(best_sol, N):
    return best_sol.copy()

def get_e_greedy_start(best_sol, N, **kwargs):
    epsilon = kwargs.get('epsilon', 0.5)

    if random.random() < epsilon:
        return np.random.randint(2, size=N)
    return best_sol.copy()

ALGORITHMS = {
    'fi': lambda sol, mod, msk: local_search(sol, get_surrogate_fitness_fn(mod, msk), first_improvement),
    'bi': lambda sol, mod, msk: local_search(sol, get_surrogate_fitness_fn(mod, msk), best_improvement),
    'tabu': lambda sol, mod, msk: tabu_search(sol, get_surrogate_fitness_fn(mod, msk))
}

START_STRATEGIES = {
    'rd': get_random_start,
    'best': get_best_start,
    'egreedy': get_e_greedy_start
}