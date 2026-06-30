import numpy as np

from src.core.fitness import get_black_box_fitness_fn


def perturb(solution, alpha):
    N = len(solution)
    result = solution.copy()
    n_flips = max(1, int(alpha * N))

    indices = np.arange(N)
    np.random.shuffle(indices)

    for i in range(n_flips):
        result[indices[i]] = 1 - result[indices[i]]

    return result


def find_connected_components(p1, p2, vig):
    N = len(p1)
    is_differing = (p1 != p2)
    visited = [False] * N
    components = []

    for start in range(N):
        if not is_differing[start] or visited[start]:
            continue

        component = []
        stack = [start]
        visited[start] = True

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in vig[current]:
                if is_differing[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        components.append(component)

    return components


def partition_crossover(p1, p2, vig, landscape, fit_p1):
    components = find_connected_components(p1, p2, vig)
    child = p1.copy()

    for component in components:
        sol_test = p1.copy()
        for i in component:
            sol_test[i] = p2[i]

        if landscape.eval(sol_test) > fit_p1:
            for i in component:
                child[i] = p2[i]

    return child


def drils_nk(landscape, vig, local_search_fn, alpha, max_iter, start_sol):
    fit_fn = get_black_box_fitness_fn(landscape)

    p1 = local_search_fn(start_sol.copy(), fit_fn)
    fit_p1 = landscape.eval(p1)
    best = p1.copy()
    best_fit = fit_p1

    for _ in range(max_iter):
        x = perturb(p1, alpha)
        p2 = local_search_fn(x, fit_fn)

        child = partition_crossover(p1, p2, vig, landscape, fit_p1)

        if np.array_equal(child, p1) or np.array_equal(child, p2):
            p1 = p2
        else:
            p1 = local_search_fn(child, fit_fn)

        fit_p1 = landscape.eval(p1)

        if fit_p1 > best_fit:
            best_fit = fit_p1
            best = p1.copy()

    return best
