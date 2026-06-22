import numpy as np

from src.core.data_sampler import generate_random_solutions, evaluate_and_transform, transform
from src.core.models import MODELS

def run_surrogate_optimization(budget, N, initial_pop_size, landscape, masks, surrogate_model, search_fun, start_fun, duplicate_strategy='ignore'):

    RETRAIN_EVERY = 1 # à tester pour gain de performance, évaluer l'impact sur la qualité du modéle

    # ajouter une fenetre sur les données d'entrainement (on remplace la liste par une queue)
    # + facteur d'exploration tabou dynamique ?

    fitness_history = []
    surrogate_fidelity_history = [] # utilise pour la courbe de fitness prédite vs réel
    alpha_history = [] # alpha sélectionné par BIC à chaque refit (uniquement pour lasso_lars_ic)
    is_lasso_lars_ic = isinstance(surrogate_model, type(MODELS['lasso_lars_ic']['model']))

    n_evals = initial_pop_size

    initial_population = generate_random_solutions(N, initial_pop_size)

    X_train, Y_train = evaluate_and_transform(masks, landscape.eval, initial_population)

    # Listes Python au lieu de numpy arrays pour accumuler les données d'entraînement :
    # np.vstack/np.append recopient tout le tableau à chaque itération (O(n) copies × budget itérations = O(n²) au total)
    # list.append est O(1) amorti — on ne convertit en numpy qu'au moment du fit
    X_list = list(X_train)
    Y_list = list(Y_train)

    best_fit_pos = np.argmax(Y_train)
    best_sol = initial_population[best_fit_pos]
    best_fit = Y_train[best_fit_pos]

    fitness_history.append((best_fit, n_evals))

    # solutions_evaluees = {tuple(sol) for sol in initial_population} # trouver un nom anglais
    visited = {tuple(sol) for sol in initial_population}

    while n_evals < budget:

        if n_evals % RETRAIN_EVERY == 0:
            surrogate_model.fit(np.array(X_list), np.array(Y_list))

            if is_lasso_lars_ic:
                alpha_history.append((n_evals, surrogate_model.alpha_))

        start_sol = start_fun(best_sol, N) # random par défaut, egreedy et best disponible

        candidate_sol = search_fun(start_sol, surrogate_model, masks) # tabu par défaut, local search bi et fi disponible

        is_random = False
        if tuple(candidate_sol) in visited:
            match duplicate_strategy:
                case 'ignore':
                    pass
                case 'skip':
                    continue
                case 'random':
                    while tuple(candidate_sol) in visited:
                        candidate_sol = np.random.randint(2, size=N)
                    is_random = True

        real_fitness = landscape.eval(candidate_sol)
        n_evals += 1

        visited.add(tuple(candidate_sol))

        candidate_walsh = transform(masks, [candidate_sol])

        X_list.append(candidate_walsh[0])
        Y_list.append(real_fitness)

        predicted_fitness = surrogate_model.predict(candidate_walsh)[0]

        print(f"Iteration : {n_evals} ; Score prédit {predicted_fitness} ; Score réel {real_fitness}")

        # En random, les derniers points évalués par le surrogate ont tous une fitness moyenne (les points aux niveaux d'itéations le plus élévés ne sont pas au zones attendues)
        # Cela peut porter confusion aux niveau d'historicité ds le graphe de fidelité de l'optimisation
        # Par conséquent si la solution est random, on entraine bien le modèle surrogate dessus mais on ne l'ajoute pas à l'historique de fidélité
        # Implémentaion peut-être un peu faiblarde, à revoir
        if not is_random: 
            surrogate_fidelity_history.append((predicted_fitness, real_fitness, n_evals))

        if real_fitness > best_fit:
            best_fit = real_fitness
            best_sol = candidate_sol
            print(f"Nouveau meilleur score : {best_fit:.6f} | Budget restant : {budget - n_evals}")

            fitness_history.append((best_fit, n_evals))

    fitness_history.append((best_fit, budget))

    return best_fit, best_sol, fitness_history, surrogate_model, surrogate_fidelity_history, alpha_history

DUPLICATE_STRATEGIES = ['ignore', 'random', 'skip']

