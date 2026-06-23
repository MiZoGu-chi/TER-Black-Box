#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class SurrogateSearcher {

public:
    enum class HillClimberStrategy {
        BEST_IMPROVEMENT,
        FIRST_IMPROVEMENT
    };

    enum class LocalSearchStrategy {
        HC_BEST_IMPROVEMENT,
        HC_FIRST_IMPROVEMENT,
        TABU
    };

    SurrogateSearcher(const std::vector<std::vector<int>>& masks, int N);

    py::array_t<int> tabu_search(py::array_t<int> start_sol, py::array_t<double> coefs, int max_iter, unsigned int seed);

    py::array_t<int> hill_climber(py::array_t<int> start_sol, py::array_t<double> coefs, HillClimberStrategy strat, unsigned int seed);

    std::vector<std::vector<int>> find_connected_components(const std::vector<bool>& is_active, const py::array_t<double>& coefs_);

    bool compare_component(const std::vector<int>& component, const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs);

    std::vector<int> partition_crossover(const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs);

    std::vector<int> perturb(const std::vector<int>& solution, double alpha, unsigned int seed);

    py::array_t<int> drils_search(py::array_t<int> start_sol, py::array_t<double> coefs, double alpha, int max_iter, LocalSearchStrategy strat, unsigned int seed, int tabu_max_iter = 0);

    void init_walsh_state(const py::array_t<int>& sol, const py::array_t<double>& coefs, std::vector<int>& current, std::vector<int>& best, std::vector<double>& phi, std::vector<double>& delta, double& cur_fit);
    
    void apply_incremental_flip(int bit, const py::array_t<double>& coefs, std::vector<int>& current, std::vector<double>& phi, std::vector<double>& delta, double& cur_fit);

    std::vector<int> to_vector(const py::array_t<int>& a);

    py::array_t<int> to_array(const std::vector<int>& v);

    double evaluate_fitness(const std::vector<int>& solution, const py::array_t<double>& coefs);

    py::array_t<int> local_search(py::array_t<int> start_sol, py::array_t<double> coefs_, LocalSearchStrategy start, int tabu_max_iter, unsigned int seed);

private:

    int _N;
    std::vector<std::vector<int>> _masks;
    std::vector<std::vector<int>> _bit_to_masks;  // précompilé une seule fois d'ou l'intêret de faire une classe, permet de retrouver les masques affecté par un changement de bits
};