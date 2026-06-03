#include "surrogate_search.hpp"

#include <pybind11/stl.h>
#include <random>

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
             py::arg("max_iter"),  py::arg("seed") = 42);
}
