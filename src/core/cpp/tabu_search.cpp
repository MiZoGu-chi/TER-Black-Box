#include "tabu_search.hpp"

#include <random>
#include <algorithm>
#include <cmath>

py::array_t<int> TabuSearch::search(py::array_t<int> start_sol, py::array_t<double> coefs_, unsigned int seed) {
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

    for (int iter = 1; iter <= _max_iter; iter++) {
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

    return to_array(best);
}
