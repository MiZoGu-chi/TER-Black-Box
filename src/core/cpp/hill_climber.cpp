#include "hill_climber.hpp"

#include <algorithm>
#include <numeric>

int HillClimberStrategyBI::select(const std::vector<double>& delta, std::mt19937&) {
    int best_bit = -1;
    double best_delta = 0.0;

    for (int b = 0; b < (int)delta.size(); b++) {
        if (delta[b] > best_delta) {
            best_delta = delta[b];
            best_bit = b;
        }
    }

    return best_bit;
}

int HillClimberStrategyFI::select(const std::vector<double>& delta, std::mt19937& rng) {
    std::vector<int> indices(delta.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int b : indices)
        if (delta[b] > 0.0) return b;

    return -1;
}

std::unique_ptr<HillClimberStrategy> HillClimber::make_strategy(Strategy s) {
    switch (s) {
        case Strategy::BEST_IMPROVEMENT:  return std::make_unique<HillClimberStrategyBI>();
        case Strategy::FIRST_IMPROVEMENT: return std::make_unique<HillClimberStrategyFI>();
    }
}

py::array_t<int> HillClimber::search(py::array_t<int> start_sol, py::array_t<double> coefs_, unsigned int seed) {
    int n_coefs = (int)_masks.size();
    std::vector<int> current(_N), best(_N);
    std::vector<double> phi(n_coefs), delta(_N);
    double cur_fit;

    init_walsh_state(start_sol, coefs_, current, best, phi, delta, cur_fit);

    std::mt19937 rng(seed);

    while (true) {
        int bit = _strategy->select(delta, rng);
        if (bit == -1) break;

        apply_incremental_flip(bit, coefs_, current, phi, delta, cur_fit);
        best = current;
    }

    return to_array(best);
}
