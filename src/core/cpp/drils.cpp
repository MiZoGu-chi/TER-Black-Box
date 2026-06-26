#include "drils.hpp"

#include <random>
#include <stdexcept>

#include "hill_climber.hpp"
#include "tabu_search.hpp"

DRILS::DRILS(const std::vector<std::vector<int>>& masks, int N, LocalSearch local_search, double alpha, int max_iter, int tabu_max_iter)
    : ILS(masks, N), _alpha(alpha), _max_iter(max_iter) {

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

py::array_t<int> DRILS::search(py::array_t<int> start_sol, py::array_t<double> coefs_, unsigned int seed) {
    std::mt19937 rng(seed);

    std::vector<int> p1 = to_vector(_local_search->search(start_sol, coefs_, rng()));
    std::vector<int> best = p1;
    double best_fit = evaluate_fitness(p1, coefs_);

    // non incrémental, à modifier plus tard (garder en mémoire la fitness de p1)
    for (int iter = 0; iter < _max_iter; iter++) {
        std::vector<int> x = perturb(p1, _alpha, rng());
        std::vector<int> p2 = to_vector(_local_search->search(to_array(x), coefs_, rng()));

        std::vector<int> child = partition_crossover(p1, p2, coefs_);

        // Le PX n'a pas trouvé de combinaison meilleure que les parents existants
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

