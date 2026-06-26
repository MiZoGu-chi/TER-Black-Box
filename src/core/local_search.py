import numpy as np

def get_hamming1_neighborhood(solution):
    N = len(solution)

    neighborhood = np.tile(solution, (N, 1)) # matrice N*N ou chaque ligne est une copie de la solution

    neighborhood[np.arange(N), np.arange(N)] = 1 - solution
    # (N = 4) neighborhood[[0,1,2,3], [0,1,2,3]] -> retourne une liste des points [neigh[0,0], neigh[1,1], neigh[2,2], neigh[3,3]] ; il suffit alors de soustraire l'inverse de la solution (1 - solution)

    return neighborhood

def tabu_search(current_best, fitness_fn):
    N = len(current_best)
    exploration_factor = 2
    max_iter = N * exploration_factor
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
        neighborhood = get_hamming1_neighborhood(current_solution)
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

                    if (np.random.randint(0, trouve + 1) == 0):
                        best_x = x

            # plus besoin de vérifier le budget

        current_fitness += best_delta
        current_solution[best_x] = 1 - current_solution[best_x]

        if (current_fitness > best_fitness):
            best_fitness = current_fitness
            best_solution = np.copy(current_solution)

        tabu_tenure[best_x] = int(params1) + np.random.randint(0, int(params2) + 1) + iter_

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

def local_search(current_best, fitness_fn, strategy_fn):
    x_star = current_best.copy()
    x_star_fitness = fitness_fn([x_star])[0]

    is_local_optimum = False

    while not is_local_optimum:

        neighborhood = get_hamming1_neighborhood(x_star)

        neighborhood_fitness = fitness_fn(neighborhood) # pas idéal pour FI car non séquentiel ? (on génére toutes les solutions d'un cooup au lieu de les parcourir une par une)

        best_idx = strategy_fn(neighborhood_fitness, x_star_fitness)

        if best_idx is not None:
            x_star = neighborhood[best_idx]
            x_star_fitness = neighborhood_fitness[best_idx]
        else:
            is_local_optimum = True

    return x_star
