import numpy as np

from src.core.data_sampler import generate_random_solutions, evaluate_and_transform, transform

def run_surrogate_optimization(budget, N, initial_pop_size, landscape, masks, surrogate_model, search_fun, start_fun):

    RETRAIN_EVERY = 1 # à tester pour gain de performance, évaluer l'impact sur la qualité du modéle

    fitness_history = []
    surrogate_fidelity_history = [] # utilise pour la courbe de fitness prédite vs réel
    
    n_evals = initial_pop_size

    initial_population = generate_random_solutions(N, initial_pop_size)

    X_train, Y_train = evaluate_and_transform(masks, landscape.eval, initial_population)


    best_fit_pos = np.argmax(Y_train) 
    best_sol = initial_population[best_fit_pos]
    best_fit = Y_train[best_fit_pos]

    fitness_history.append((best_fit, n_evals))

    # solutions_evaluees = {tuple(sol) for sol in initial_population} # trouver un nom anglais

    while n_evals < budget:

        if n_evals % RETRAIN_EVERY == 0:
            #print(X_train.shape)
            #print(Y_train.shape)
            #print(X_train)
            #print(Y_train)
            surrogate_model.fit(X_train, Y_train)

        start_sol = start_fun(best_sol, N) # random par défaut, egreedy et best disponible

        candidate_sol = search_fun(start_sol, surrogate_model, masks) # tabu par défaut, local search bi et fi disponible

        # MODIFICATION IMPORTANTE : on ne réavalue pas / n'étraine pas sur une solution déjà évaluée
        #while tuple(candidate_sol) in solutions_evaluees:  # while si jamais on génére aléatoirement une solution déjà évalué meme si ça parait impropable ... 
            #candidate_sol = np.random.randint(2, size=N)  

        # if tuple(candidate_sol) in solutions_evaluees:   # tanpis si on retrouve le meme point
        #     continue
        
        # solutions_evaluees.add(tuple(candidate_sol))

        real_fitness = landscape.eval(candidate_sol) 
        n_evals += 1

        candidate_walsh = transform(masks, [candidate_sol])

        X_train = np.vstack([X_train, candidate_walsh])
        Y_train = np.append(Y_train, real_fitness)

        predicted_fitness = surrogate_model.predict(candidate_walsh)[0]

        print(f"Iteration : {n_evals} ; Score prédit {predicted_fitness} ; Score réel {real_fitness}")
        surrogate_fidelity_history.append((predicted_fitness, real_fitness, n_evals))

        if real_fitness > best_fit:
            best_fit = real_fitness
            best_sol = candidate_sol
            print(f"Nouveau meilleur score : {best_fit:.6f} | Budget restant : {budget - n_evals}")

            fitness_history.append((best_fit, n_evals))

    fitness_history.append((best_fit, budget))

    return best_fit, best_sol, fitness_history, surrogate_model, surrogate_fidelity_history

