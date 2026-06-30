#include "drils.hpp"

#include <random>
#include <algorithm>
#include <numeric>

#include "hill_climber.hpp"
#include "tabu_search.hpp"

DRILSBase::DRILSBase(const std::vector<std::vector<int>>& masks, int N, LocalSearch local_search, double alpha, int max_iter, int tabu_max_iter)
    : WalshSurrogateAlgorithm(masks, N), _alpha(alpha), _max_iter(max_iter) {

    switch (local_search) {
        case LocalSearch::HC_BEST_IMPROVEMENT:
            _local_search = std::make_unique<HillClimber>(masks, N, _bit_to_masks, HillClimber::make_strategy(HillClimber::Strategy::BEST_IMPROVEMENT));
            break;
        case LocalSearch::HC_FIRST_IMPROVEMENT:
            _local_search = std::make_unique<HillClimber>(masks, N, _bit_to_masks, HillClimber::make_strategy(HillClimber::Strategy::FIRST_IMPROVEMENT));
            break;
        case LocalSearch::TABU:
            _local_search = std::make_unique<TabuSearch>(masks, N, _bit_to_masks, tabu_max_iter);
            break;
    }
}

std::vector<int> DRILSBase::perturb(const std::vector<int>& solution, double alpha, unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<int> result = solution;

    // !! pour un N petit et/ou un alpha faible, (int)(alpha * _N) peut tronquer à 0 !!
    int n_flips = (int)(alpha * _N);

    std::vector<int> indices(_N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < n_flips; i++)
        result[indices[i]] = 1 - result[indices[i]];

    return result;
}

std::vector<std::vector<int>> DRILSBase::find_connected_components(const std::vector<bool>& is_differing, const py::array_t<double>& coefs_) {
    auto coefs = coefs_.unchecked<1>();
    std::vector<std::vector<int>> components;
    std::vector<bool> visited(_N, false);

    for (int start = 0; start < _N; start++) {
        if (!is_differing[start] || visited[start]) continue;

        std::vector<int> component;
        std::vector<int> stack = {start};
        visited[start] = true;

        while (!stack.empty()) {
            int current = stack.back();
            stack.pop_back();
            component.push_back(current);

            for (int mask_idx : _bit_to_masks[current]) {
                if (coefs(mask_idx) == 0.0) continue;

                for (int neighbor : _masks[mask_idx]) {
                    if (neighbor == current) continue;
                    if (!is_differing[neighbor]) continue;
                    if (visited[neighbor]) continue;

                    visited[neighbor] = true;
                    stack.push_back(neighbor);
                }
            }
        }

        components.push_back(component);
    }

    return components;
}

bool DRILSBase::compare_component(const std::vector<int>& component, const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs_) {
    auto coefs = coefs_.unchecked<1>();

    std::vector<bool> mask_seen(_masks.size(), false);
    std::vector<int> relevant_masks;

    for (int var : component) {
        for (int mask_idx : _bit_to_masks[var]) {
            if (!mask_seen[mask_idx]) {
                mask_seen[mask_idx] = true;
                relevant_masks.push_back(mask_idx);
            }
        }
    }

    double g_p1 = 0.0, g_p2 = 0.0;

    for (int mask_idx : relevant_masks) {
        double c = coefs(mask_idx);
        if (c == 0.0) continue;

        int sum_p1 = 0, sum_p2 = 0;
        for (int b : _masks[mask_idx]) {
            sum_p1 += p1[b];
            sum_p2 += p2[b];
        }

        double phi_p1 = (sum_p1 % 2 == 0) ? 1.0 : -1.0;
        double phi_p2 = (sum_p2 % 2 == 0) ? 1.0 : -1.0;

        g_p1 += c * phi_p1;
        g_p2 += c * phi_p2;
    }

    return g_p1 > g_p2;
}

std::vector<int> DRILSBase::partition_crossover(const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs_) {
    std::vector<bool> is_differing(_N);

    for (int i = 0; i < _N; i++)
        is_differing[i] = (p1[i] != p2[i]);

    auto components = find_connected_components(is_differing, coefs_);

    std::vector<int> child = p1;

    for (const auto& component : components) {
        bool p1_greater = compare_component(component, p1, p2, coefs_);
        if (!p1_greater) {
            for (int var : component)
                child[var] = p2[var];
        }
    }

    return child;
}




py::array_t<int> DRILS::search(py::array_t<int> start_sol, py::array_t<double> coefs_, unsigned int seed) {
    std::mt19937 rng(seed);

    std::vector<int> p1 = to_vector(_local_search->search(start_sol, coefs_, rng()));
    std::vector<int> best = p1;
    double best_fit = evaluate_fitness(p1, coefs_);

    for (int iter = 0; iter < _max_iter; iter++) {
        std::vector<int> x = perturb(p1, _alpha, rng());
        std::vector<int> p2 = to_vector(_local_search->search(to_array(x), coefs_, rng()));

        std::vector<int> child = partition_crossover(p1, p2, coefs_);

        if (child == p1 || child == p2) {
            p1 = p2;
            // on change de bassin d'attraction (non-elitist design of the acceptance criterion)
        } else {
            // le PX a trouvé une meilleure combinaison ; on applique le LS pour garantir
            // que p1 est un optimum local (requis pour que le PX soit efficace à la prochaine itération)
            p1 = to_vector(_local_search->search(to_array(child), coefs_, rng()));
        }

        double fit = evaluate_fitness(p1, coefs_);

        if (fit > best_fit) {
            best_fit = fit;
            best = p1;
        }
    }

    return to_array(best);
}




DRILSPlus::DRILSPlus(const std::vector<std::vector<int>>& masks, int N, DRILSBase::LocalSearch local_search, double alpha, double beta, int max_iter, int tabu_max_iter)
    : DRILSBase(masks, N, local_search, alpha, max_iter, tabu_max_iter), _beta(beta) {}

py::array_t<int> DRILSPlus::search(py::array_t<int> start_sol, py::array_t<double> coefs_, unsigned int seed) {
    std::mt19937 rng(seed);

    std::vector<int> p1 = to_vector(_local_search->search(start_sol, coefs_, rng()));
    std::vector<int> best = p1;
    double best_fit = evaluate_fitness(p1, coefs_);

    for (int iter = 0; iter < _max_iter; iter++) {
        std::vector<int> x = perturb(p1, _alpha, rng());
        std::vector<int> p2 = to_vector(_local_search->search(to_array(x), coefs_, rng()));

        std::vector<int> child = partition_crossover(p1, p2, coefs_);

        if (child == p1 || child == p2) {
            // On repart de p1 avec _beta au lieu de _alpha car
            // _alpha est calibré pour PX (p2 proche de p1) et reconvergerait vers
            // le même bassin. _beta permet de s'échapper de ce bassin
            x = perturb(p1, _beta, rng());
            p1 = to_vector(_local_search->search(to_array(x), coefs_, rng()));
        } else {
            // le PX a trouvé une meilleure combinaison ; on applique le LS pour garantir
            // que p1 est un optimum local (requis pour que le PX soit efficace à la prochaine itération)
            p1 = to_vector(_local_search->search(to_array(child), coefs_, rng()));
        }

        double fit = evaluate_fitness(p1, coefs_);

        if (fit > best_fit) {
            best_fit = fit;
            best = p1;
        }
    }

    return to_array(best);
}

DRILSPlusPlus::DRILSPlusPlus(const std::vector<std::vector<int>>& masks, int N, DRILSBase::LocalSearch local_search, double alpha, double beta, double gamma, int max_iter, int tabu_max_iter)
    : DRILSBase(masks, N, local_search, alpha, max_iter, tabu_max_iter), _beta(beta), _gamma(gamma) {}

py::array_t<int> DRILSPlusPlus::search(py::array_t<int> start_sol, py::array_t<double> coefs_, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<int> p1 = to_vector(_local_search->search(start_sol, coefs_, rng()));
    std::vector<int> best = p1;
    double best_fit = evaluate_fitness(p1, coefs_);

    for (int iter = 0; iter < _max_iter; iter++) {
        if (dist(rng) < _gamma) {
            // on renouvelle p1 sans crossover pour explorer
            // un nouveau bassin et alimenter le PX avec un parent plus frais
            std::vector<int> x = perturb(p1, _beta, rng());
            p1 = to_vector(_local_search->search(to_array(x), coefs_, rng()));
        } else {
            std::vector<int> x = perturb(p1, _alpha, rng());
            std::vector<int> p2 = to_vector(_local_search->search(to_array(x), coefs_, rng()));

            std::vector<int> child = partition_crossover(p1, p2, coefs_);

            if (child == p1 || child == p2) {
                p1 = p2;
            } else {
                p1 = to_vector(_local_search->search(to_array(child), coefs_, rng()));
            }
        }

        double fit = evaluate_fitness(p1, coefs_);

        if (fit > best_fit) {
            best_fit = fit;
            best = p1;
        }
    }

    return to_array(best);
}