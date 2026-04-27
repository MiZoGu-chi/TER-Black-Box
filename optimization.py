import numpy as np

np.set_printoptions(threshold=256)

from data_sampler import generate_random_solutions, evaluate_and_transform, transform

from walsh import apply_walsh

from plot import plot_convergence_comparison

from algorithm import get_black_box_fitness_fn, local_search, best_improvement, first_improvement, get_random_start, get_best_start, get_e_greedy_start

def run_surrogate_optimization(args, landscape, masks, surrogate_model, search_fun, start_fun):

    best_scores = []

    budget = args.budget
    #window_size = args.window_size
    
    print(f"Lancement avec budget {budget} ...")

    budget -= args.initial_pop_size

    initial_population = generate_random_solutions(args.N, args.initial_pop_size)

    print(f"Population initiale :\n {initial_population}")

    X_train, Y_train = evaluate_and_transform(masks, landscape.eval, initial_population)

    best_sol_pos = np.argmax(Y_train) 
    best_sol = initial_population[best_sol_pos]
    best_score = Y_train[best_sol_pos]

    temp_best = -np.inf

    for s in Y_train:
        if s > temp_best: temp_best = s
        best_scores.append(temp_best)

    print(f"Meilleure fitness sur la population initiale : {best_score} ; pour la solution {best_sol}")

    while budget > 0: 
        surrogate_model.fit(X_train, Y_train) 

        start_sol = start_fun(best_sol, args.N)

        candidate_sol = search_fun(start_sol, surrogate_model, masks)

        candidate_score = landscape.eval(candidate_sol) 
        budget -= 1

        candidate_walsh = transform(masks, [candidate_sol])
        X_train = np.vstack([X_train, candidate_walsh])
        Y_train = np.append(Y_train, candidate_score)

        if candidate_score > best_score:
            best_score = candidate_score
            best_sol = candidate_sol
            print(f"Nouveau meilleur score : {best_score:.6f} | Budget restant : {budget}")
        else: 
            print(f"Budget restant : {budget}")

        best_scores.append(best_score)

    baseline = get_baseline_score(landscape, n_runs=10)
    print(f"Baseline calculée : {baseline:.6f}")

    plot_convergence_comparison(best_scores, baseline)


def get_baseline_score(landscape, n_runs=30):
    scores = [] 
    
    fit_fn = get_black_box_fitness_fn(landscape)
    
    for _ in range(n_runs):
        start_sol = np.random.randint(2, size=landscape.N)

        optimum_local_sol = local_search(start_sol, fit_fn, best_improvement) 
        
        scores.append(landscape.eval(optimum_local_sol))
        
    return np.mean(scores)
