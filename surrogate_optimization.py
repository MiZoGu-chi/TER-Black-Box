import numpy as np

np.set_printoptions(threshold=256)

from data_sampler import generate_random_solutions, evaluate_and_transform, transform

from walsh import apply_walsh

from algorithm import get_black_box_fitness_fn, local_search, best_improvement, first_improvement, get_random_start, get_best_start, get_e_greedy_start

def run_surrogate_optimization(budget, N, initial_pop_size, landscape, masks, surrogate_model, search_fun, start_fun):

    fitness_history = []
    surrogate_fidelity_history = [] # utilise pour la courbe de fitness prédite vs réel

    #window_size = args.window_size
    
    n_evals = initial_pop_size

    initial_population = generate_random_solutions(N, initial_pop_size)

    X_train, Y_train = evaluate_and_transform(masks, landscape.eval, initial_population)


    best_fit_pos = np.argmax(Y_train) 
    best_sol = initial_population[best_fit_pos]
    best_fit = Y_train[best_fit_pos]

    fitness_history.append((best_fit, n_evals))

    while n_evals < budget: 
        surrogate_model.fit(X_train, Y_train) 

        start_sol = start_fun(best_sol, N) # random par défaut, egreedy et best disponible

        candidate_sol = search_fun(start_sol, surrogate_model, masks) # tabu par défaut, local search bi et fi disponible

        real_fitness = landscape.eval(candidate_sol) 
        n_evals += 1

        candidate_walsh = transform(masks, [candidate_sol])

        X_train = np.vstack([X_train, candidate_walsh])
        Y_train = np.append(Y_train, real_fitness)

        predicted_fitness = surrogate_model.predict(candidate_walsh)[0]

        if n_evals % 10 == 0:
            print(f"Iteration : {n_evals} ; Score prédit {predicted_fitness} ; Score réel {real_fitness}")
            surrogate_fidelity_history.append((predicted_fitness, real_fitness, n_evals))

        if real_fitness > best_fit:
            best_fit = real_fitness
            best_sol = candidate_sol
            print(f"Nouveau meilleur score : {best_fit:.6f} | Budget restant : {budget - n_evals}")

            fitness_history.append((best_fit, n_evals))

    fitness_history.append((best_fit, budget))

    return best_fit, best_sol, fitness_history, surrogate_model, surrogate_fidelity_history


def get_baseline_score(landscape, n_runs=30):
    scores = [] 
    
    fit_fn = get_black_box_fitness_fn(landscape)
    
    for _ in range(n_runs):
        start_sol = np.random.randint(2, size=landscape.N)

        optimum_local_sol = local_search(start_sol, fit_fn, best_improvement) 
        
        scores.append(landscape.eval(optimum_local_sol))
        
    return np.mean(scores)
