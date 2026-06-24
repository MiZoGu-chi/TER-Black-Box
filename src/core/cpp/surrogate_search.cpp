#include "surrogate_search.hpp"

#include <pybind11/stl.h>
#include <random>
#include <algorithm>
#include <numeric>

SurrogateSearcher::SurrogateSearcher(const std::vector<std::vector<int>>& masks, int N)
        : _masks(masks), _N(N), _bit_to_masks(N) {
        for (int i = 0; i < (int)_masks.size(); i++)
            for (int b : _masks[i])
                _bit_to_masks[b].push_back(i);
}

py::array_t<int> SurrogateSearcher::local_search(py::array_t<int> start_sol, py::array_t<double> coefs_, LocalSearchStrategy strat, int tabu_max_iter, unsigned int seed) {
    switch (strat) {
        case LocalSearchStrategy::HC_FIRST_IMPROVEMENT:
            return hill_climber(start_sol, coefs_, HillClimberStrategy::FIRST_IMPROVEMENT, seed);
        case LocalSearchStrategy::HC_BEST_IMPROVEMENT:
            return hill_climber(start_sol, coefs_, HillClimberStrategy::BEST_IMPROVEMENT, seed);
        case LocalSearchStrategy::TABU:
            return tabu_search(start_sol, coefs_, tabu_max_iter, seed);
    }
}

void SurrogateSearcher::init_walsh_state(const py::array_t<int>& sol_, const py::array_t<double>& coefs_, std::vector<int>& current, std::vector<int>& best, std::vector<double>& phi, std::vector<double>& delta, double& cur_fit) {
    // On utilise unchecked() au lieu de std::vector pour lire la ram de Python directement sans faire de copie mémoire, plus rapide
    // type -> pybind11::detail::unchecked_reference<int/double, 1> ; (tableau à 1 dimension) ; lecture seule
    auto sol  = sol_.unchecked<1>();
    auto coefs = coefs_.unchecked<1>();

    int n_coefs = (int)_masks.size();

    for (int i = 0; i < _N; i++)
        current[i] = sol(i);

    best = current;

    // valeur de la fonction de Walsh pour le masque i sur current / phi[i] = (-1)^(somme des bits de current couverts par le masque i)
    for (int i = 0; i < n_coefs; i++) {
        int s = 0;
        for (int b : _masks[i])
            s += current[b];
        phi[i] = (s % 2 == 0) ? 1.0 : -1.0;
    }

    // fitness de la solution courante
    cur_fit = 0.0;

    for (int i = 0; i < n_coefs; i++)
        cur_fit += coefs(i) * phi[i];

    // delta[x] = gain prédit si on flippe le bit x ; à factoriser pour les différents algo ?
    for (int x = 0; x < _N; x++)
        for (int i : _bit_to_masks[x])
            delta[x] -= 2.0 * coefs(i) * phi[i];
}

void SurrogateSearcher::apply_incremental_flip(int bit, const py::array_t<double>& coefs_, std::vector<int>& current, std::vector<double>& phi, std::vector<double>& delta, double& cur_fit) {
    auto coefs = coefs_.unchecked<1>();

    current[bit] = 1 - current[bit]; // flip
    cur_fit += delta[bit]; // évalution de la nouvelle fitness

    for (int i : _bit_to_masks[bit]) { // parcouts tous les masks affecté par le bit changé
        double old_phi = phi[i];
        phi[i] = -phi[i]; // tous les masques concerné change de parité
        double upd = 4.0 * coefs(i) * old_phi; // gain / perte ; ajouter la formule mathématique pour la clarté

        for (int y : _masks[i]) {
            delta[y] += upd;
        }
    }
}

py::array_t<int> SurrogateSearcher::hill_climber(py::array_t<int> start_sol, py::array_t<double> coefs_, HillClimberStrategy strat, unsigned int seed) {
    int n_coefs = (int)_masks.size();

    std::vector<int> current(_N), best(_N);
    std::vector<double> phi(n_coefs), delta(_N);
    double cur_fit;

    init_walsh_state(start_sol, coefs_, current, best, phi, delta, cur_fit);

    std::mt19937 rng(seed);
    bool improved = true;


    while (improved) {
        improved = false;
        int best_bit_to_flip = -1;

        switch (strat) {
            case HillClimberStrategy::BEST_IMPROVEMENT: {
                double best_delta = 0.0;

                // boucle simple pour trouver le bit qui donne la meilleur fitness
                for (int b = 0; b < _N; b++) {
                    if (delta[b] > best_delta) {
                        best_delta = delta[b];
                        best_bit_to_flip = b;
                        improved = true;
                    }
                }
                break;
            }
            case HillClimberStrategy::FIRST_IMPROVEMENT: {
                std::vector<int> indices(_N);
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), rng);

                for (int b : indices) {
                    if (delta[b] > 0.0) {
                        best_bit_to_flip = b;
                        improved = true;
                        break;
                    }
                }
                break;
            }
        }

        if (improved) {
            apply_incremental_flip(best_bit_to_flip, coefs_, current, phi, delta, cur_fit);
            best = current;
        }

    }

    py::array_t<int> result(_N);
    auto r = result.mutable_unchecked<1>();
    for (int i = 0; i < _N; i++) r(i) = best[i];
    return result;
}

std::vector<std::vector<int>> SurrogateSearcher::find_connected_components(const std::vector<bool>& is_active, const py::array_t<double>& coefs_) {
    auto coefs = coefs_.unchecked<1>();
    std::vector<std::vector<int>> components;
    std::vector<bool> visited(_N, false);

    // DFS
    for (int start = 0; start < _N; start++) {
        if (!is_active[start] || visited[start]) continue;

        std::vector<int> component;
        std::vector<int> stack = {start};

        visited[start] = true;

        while (!stack.empty()) {
            int current = stack.back();
            stack.pop_back();
            component.push_back(current);

            // voisins de current = toutes les variables qui partagent
            // un masque actif avec current
            for (int mask_idx : _bit_to_masks[current]) {
                if (coefs(mask_idx) == 0.0) continue;  // masque inactif

                for (int neighbor : _masks[mask_idx]) {
                    if (neighbor == current) continue;
                    if (!is_active[neighbor]) continue;  // bit commun -> ignoré
                    if (visited[neighbor]) continue;

                    visited[neighbor] = true;
                    stack.push_back(neighbor);
                }
            }
        }

        components.push_back(component);
    } 

    // components = { C_1, C_2, ..., C_q }
    // C_i = { x_i1, x_i2, ..., x_ij } : i-ème composante connexe,
    // contient les variables actives connectées entre elles via les masques actifs
    // partition de { j | is_active[j] } en q composantes connexes indépendantes

    return components;
}

bool SurrogateSearcher::compare_component(const std::vector<int>& component, const std::vector<int>& p1,  const std::vector<int>& p2, py::array_t<double> coefs_) {
    auto coefs = coefs_.unchecked<1>();

    std::vector<bool> mask_seen(_masks.size(), false);
    std::vector<int> relevant_masks;

    for (int var : component) {
        for (int mask_idx : _bit_to_masks[var]) {
            if (!mask_seen[mask_idx]) {
                mask_seen[mask_idx] = true; // si plusieurs var du componet sont liées à un meme masque on ne l'ajoute pas deux fois
                relevant_masks.push_back(mask_idx);
            }
        }
    }

    double g_p1 = 0.0, g_p2 = 0.0;

    for (int mask_idx : relevant_masks) { // parcours les masques 
        double c = coefs(mask_idx);
        if (c == 0.0) continue;

        int sum_p1 = 0, sum_p2 = 0;
        for (int b : _masks[mask_idx]) { // parcours les bits à 1 du masque
            sum_p1 += p1[b]; // somme des bits de p1 que regarde le masque
            sum_p2 += p2[b]; // ..
        }

        double phi_p1 = (sum_p1 % 2 == 0) ? 1.0 : -1.0;
        double phi_p2 = (sum_p2 % 2 == 0) ? 1.0 : -1.0;

        g_p1 += c * phi_p1;
        g_p2 += c * phi_p2;
    }

    return g_p1 > g_p2;
}

std::vector<int> SurrogateSearcher::partition_crossover(const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs_) {
    std::vector<bool> is_active(_N);

    for (int i = 0; i < _N; i++)
        is_active[i] = (p1[i] != p2[i]); // on prends uniquement les bits qui différent entre les deux optima locaux

    auto components = find_connected_components(is_active, coefs_);

    // les bits communs entre p1 et p2 doivent être hérités
    std::vector<int> child = p1; // ou = p2, équivalent

    // parcours est comparaison de toutes les composantes connexes
    for (const auto& component : components) {
        bool p1_greater = compare_component(component, p1, p2, coefs_);
        if (!p1_greater) {
            for (int var : component)
                child[var] = p2[var];
        }
    }

    return child;
}

std::vector<int> SurrogateSearcher::perturb(const std::vector<int>& solution, double alpha, unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<int> result = solution;

    // !! pour un N petit et/ou un alpha faible, (int)(alpha * _N) peut tronquer à 0 !!
    int n_flips = (int)(alpha * _N);

    std::vector<int> indices(_N);
    std::iota(indices.begin(), indices.end(), 0);   // remplit avec 0, 1, ..., N-1

    std::shuffle(indices.begin(), indices.end(), rng); 

    // On inverse les n_flips premiers bits ordonnés alétoirement
    for (int i = 0; i < n_flips; i++) {
        result[indices[i]] = 1 - result[indices[i]];
    }

    return result;
}

std::vector<int> SurrogateSearcher::to_vector(const py::array_t<int>& a) {
    auto acc = a.unchecked<1>();
    std::vector<int> v(acc.shape(0));
    for (int i = 0; i < (int)v.size(); i++) v[i] = acc(i);
    return v;
}

py::array_t<int> SurrogateSearcher::to_array(const std::vector<int>& v) {
    py::array_t<int> result(v.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); i++) r(i) = v[i];
    return result;
}

double SurrogateSearcher::evaluate_fitness(const std::vector<int>& solution, const py::array_t<double>& coefs_) {
    auto coefs = coefs_.unchecked<1>();
    double fit = 0.0;

    for (int i = 0; i < (int)_masks.size(); i++) {
        if (coefs(i) == 0.0) continue;

        int s = 0;
        for (int b : _masks[i]) s += solution[b];

        fit += coefs(i) * ((s % 2 == 0) ? 1.0 : -1.0);
    }

    return fit;
}

py::array_t<int> SurrogateSearcher::drils_search(py::array_t<int> start_sol, py::array_t<double> coefs_, double alpha, int max_iter, LocalSearchStrategy strat, unsigned int seed, int tabu_max_iter) {
    std::mt19937 rng(seed);

    std::vector<int> p1 = to_vector(local_search(start_sol, coefs_, strat, tabu_max_iter, rng()));
    std::vector<int> best = p1;
    double best_fit = evaluate_fitness(p1, coefs_);


    // non incrémental, à modifier plus tard (garder en mémoire la fitness de p1)
    for (int iter = 0; iter < max_iter; iter++) {
        std::vector<int> x  = perturb(p1, alpha, rng());
        std::vector<int> p2 = to_vector(local_search(to_array(x), coefs_, strat, tabu_max_iter, rng()));

        std::vector<int> child = partition_crossover(p1, p2, coefs_);

        // Le PX n'a pas trouvé de combinaison meilleure que les parents existants
        if (child == p1 || child == p2) {
            p1 = p2;
            // on change de la bassin d'attraction (non-elitist design of the acceptance criterion)
            // cad même si f_hat(p2) < f_hat(p1)
        } else {
        // le PX a trouvé une meilleure combinaison
        // on applique le HC pour garantir que p1 est un optimum local
        // (requis pour que le PX soit efficace à la prochaine itération)
        // En effet child est optimal dans le sous-espace des bits libres
        // mais pas nécessairement dans l'espace complet
            p1 = to_vector(local_search(to_array(child), coefs_, strat, tabu_max_iter, rng()));
        }

        double fit = evaluate_fitness(p1, coefs_);
        
        if (fit > best_fit) {
            best_fit = fit;
            best = p1;
        }
    }

    return to_array(best);
}

py::array_t<int> SurrogateSearcher::tabu_search(py::array_t<int> start_sol, py::array_t<double> coefs_, int max_iter, unsigned int seed) {
    int n_coefs = (int)_masks.size();

    std::vector<int> current(_N), best(_N);
    std::vector<double> phi(n_coefs), delta(_N);
    double cur_fit;

    init_walsh_state(start_sol, coefs_, current, best, phi, delta, cur_fit);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> rand_tenure(0, 9);
    int tenure_base = std::max(1, _N / 10);
    std::vector<int> tabu_tenure(_N, 0);
    double best_fit = cur_fit;

    for (int iter = 1; iter <= max_iter; iter++) {
        double best_d = -1e18;
        int best_bit = -1;
        int ties = 1;

        for (int b = 0; b < _N; b++) {
            bool is_tabu = (tabu_tenure[b] >= iter);
            bool aspire = (cur_fit + delta[b] > best_fit);

            if (!is_tabu || aspire) {
                if (delta[b] > best_d + 1e-12) {
                    best_d = delta[b];
                    best_bit = b;
                    ties = 1;
                } else if (std::abs(delta[b] - best_d) <= 1e-12) {
                    if (std::uniform_int_distribution<int>(0, ties++)(rng) == 0)
                        best_bit = b;
                }
            }
        }

        if (best_bit == -1) break;

        apply_incremental_flip(best_bit, coefs_, current, phi, delta, cur_fit);

        if (cur_fit > best_fit) {
            best_fit = cur_fit;
            best = current;
        }

        tabu_tenure[best_bit] = iter + tenure_base + rand_tenure(rng);
    }

    py::array_t<int> result(_N);
    auto r = result.mutable_unchecked<1>();
    for (int i = 0; i < _N; i++) r(i) = best[i];
    return result;
}


PYBIND11_MODULE(surrogate_search, m) {
    py::enum_<SurrogateSearcher::HillClimberStrategy>(m, "HillClimberStrategy")
        .value("BEST_IMPROVEMENT", SurrogateSearcher::HillClimberStrategy::BEST_IMPROVEMENT)
        .value("FIRST_IMPROVEMENT", SurrogateSearcher::HillClimberStrategy::FIRST_IMPROVEMENT);

    py::enum_<SurrogateSearcher::LocalSearchStrategy>(m, "LocalSearchStrategy")
        .value("HC_BEST_IMPROVEMENT", SurrogateSearcher::LocalSearchStrategy::HC_BEST_IMPROVEMENT)
        .value("HC_FIRST_IMPROVEMENT", SurrogateSearcher::LocalSearchStrategy::HC_FIRST_IMPROVEMENT)
        .value("TABU", SurrogateSearcher::LocalSearchStrategy::TABU);

    py::class_<SurrogateSearcher>(m, "SurrogateSearcher")
        .def(py::init<const std::vector<std::vector<int>>&, int>(),
             py::arg("masks"), py::arg("N"))
        .def("hill_climber", &SurrogateSearcher::hill_climber,
             py::arg("start_sol"), py::arg("coefs"),
             py::arg("strat") = SurrogateSearcher::HillClimberStrategy::BEST_IMPROVEMENT,
             py::arg("seed") = 42)
        .def("tabu_search",  &SurrogateSearcher::tabu_search,
             py::arg("start_sol"), py::arg("coefs"),
             py::arg("max_iter"),  py::arg("seed") = 42)
        .def("find_connected_components", &SurrogateSearcher::find_connected_components,
            py::arg("is_active"), py::arg("coefs"))
        .def("compare_component", &SurrogateSearcher::compare_component,
             py::arg("component"), py::arg("p1"), py::arg("p2"), py::arg("coefs"))
        .def("partition_crossover", &SurrogateSearcher::partition_crossover,
             py::arg("p1"), py::arg("p2"), py::arg("coefs"))
        .def("perturb", &SurrogateSearcher::perturb,
             py::arg("solution"), py::arg("alpha"), py::arg("seed") = 42)
        .def("local_search", &SurrogateSearcher::local_search,
             py::arg("start_sol"), py::arg("coefs"), py::arg("strat"),
             py::arg("tabu_max_iter") = 0, py::arg("seed") = 42)
        .def("drils_search", &SurrogateSearcher::drils_search,
             py::arg("start_sol"), py::arg("coefs"), py::arg("alpha"),
             py::arg("max_iter"), py::arg("strat"), py::arg("seed") = 42,
             py::arg("tabu_max_iter") = 0);
}
