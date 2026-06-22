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

py::array_t<int> SurrogateSearcher::hill_climber(py::array_t<int> start_sol, py::array_t<double> coefs_) {
    int n_coefs = (int)_masks.size();

    std::vector<int> current(_N), best(_N);
    std::vector<double> phi(n_coefs), delta(_N);
    double cur_fit;

    init_walsh_state(start_sol, coefs_, current, best, phi, delta, cur_fit);

    bool improved = true;

    while (improved) {
        improved  = false;
        int best_bit_to_flip = -1;
        double best_delta = 0.0;

        // Boucle simple pour trouver le bit qui donne le meilleur gain de fitness
        for (int b = 0; b < _N; b++) {
            if (delta[b] >best_delta) {
                best_delta   = delta[b];
                best_bit_to_flip = b;
                improved = true;
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

std::vector<std::vector<int>> SurrogateSearcher::get_variable_interaction_graph(py::array_t<double> coefs_) {
    auto coefs = coefs_.unchecked<1>();
    std::vector<std::vector<int>> graph(_N);

    for (int i = 0; i < _masks.size(); i++) {
        if (coefs(i) == 0.0) continue; 

        const auto& interaction = _masks[i];

        for (int a = 0; a < interaction.size(); a++) {
            int var_a = interaction[a];

            for (int b = a + 1; b < interaction.size(); b++) {
                int var_b = interaction[b];

                graph[var_a].push_back(var_b);
                graph[var_b].push_back(var_a);
            }
        }
    }

    return graph;
}

std::vector<std::vector<int>> SurrogateSearcher::find_connected_components(const std::vector<std::vector<int>>& graph, const std::vector<bool>& is_active) {
    std::vector<std::vector<int>> components;
    std::vector<bool> visited(_N, false);

    for (int start = 0; start < _N; start++) {
        if (!is_active[start] || visited[start]) continue;

        std::vector<int> component;
        std::vector<int> stack = {start};
        visited[start] = true;

        while (!stack.empty()) {
            int current = stack.back();
            stack.pop_back();
            component.push_back(current);

            for (int neighbor : graph[current]) {
                if (is_active[neighbor] && !visited[neighbor]) {
                    visited[neighbor] = true;
                    stack.push_back(neighbor);
                }
            }
        }

        components.push_back(component);
    }

    return components;
}

bool SurrogateSearcher::compare_component(
    const std::vector<int>& component,
    const std::vector<int>& p1,
    const std::vector<int>& p2,
    py::array_t<double> coefs_)
{
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

std::vector<int> SurrogateSearcher::partition_crossover(
    const std::vector<int>& p1,
    const std::vector<int>& p2,
    py::array_t<double> coefs_)
{
    std::vector<bool> is_active(_N);
    for (int i = 0; i < _N; i++)
        is_active[i] = (p1[i] != p2[i]);

    auto graph = get_variable_interaction_graph(coefs_);
    auto components = find_connected_components(graph, is_active);

    std::vector<int> child = p1;

    for (const auto& component : components) {
        bool use_p1 = compare_component(component, p1, p2, coefs_);
        if (!use_p1) {
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

py::array_t<int> SurrogateSearcher::drils_search(py::array_t<int> start_sol, py::array_t<double> coefs_, double alpha, int max_iter, unsigned int seed) {
    std::mt19937 rng(seed);

    std::vector<int> p1 = to_vector(hill_climber(start_sol, coefs_));
    std::vector<int> best = p1;
    double best_fit = evaluate_fitness(p1, coefs_);



    for (int iter = 0; iter < max_iter; iter++) {
        std::vector<int> x  = perturb(p1, alpha, rng());
        std::vector<int> p2 = to_vector(hill_climber(to_array(x), coefs_));

        std::vector<int> child = partition_crossover(p1, p2, coefs_);

        if (child == p1 || child == p2) {
            p1 = p2;
        } else {
            p1 = to_vector(hill_climber(to_array(child), coefs_));
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
    py::class_<SurrogateSearcher>(m, "SurrogateSearcher")
        .def(py::init<const std::vector<std::vector<int>>&, int>(),
             py::arg("masks"), py::arg("N"))
        .def("hill_climber", &SurrogateSearcher::hill_climber,
             py::arg("start_sol"), py::arg("coefs"))
        .def("tabu_search",  &SurrogateSearcher::tabu_search,
             py::arg("start_sol"), py::arg("coefs"),
             py::arg("max_iter"),  py::arg("seed") = 42)
        .def("get_variable_interaction_graph", &SurrogateSearcher::get_variable_interaction_graph,
             py::arg("coefs"))
        .def("find_connected_components", &SurrogateSearcher::find_connected_components,
             py::arg("graph"), py::arg("is_active"))
        .def("compare_component", &SurrogateSearcher::compare_component,
             py::arg("component"), py::arg("p1"), py::arg("p2"), py::arg("coefs"))
        .def("partition_crossover", &SurrogateSearcher::partition_crossover,
             py::arg("p1"), py::arg("p2"), py::arg("coefs"))
        .def("perturb", &SurrogateSearcher::perturb,
             py::arg("solution"), py::arg("alpha"), py::arg("seed") = 42)
        .def("drils_search", &SurrogateSearcher::drils_search,
             py::arg("start_sol"), py::arg("coefs"), py::arg("alpha"),
             py::arg("max_iter"), py::arg("seed") = 42);
}
