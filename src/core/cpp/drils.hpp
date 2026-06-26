#pragma once

#include <memory>
#include "ils.hpp"
#include "local_search.hpp"

class DRILS : public ILS {
public:
    enum class LocalSearch { HC_BEST_IMPROVEMENT, HC_FIRST_IMPROVEMENT, TABU };

    DRILS(const std::vector<std::vector<int>>& masks, int N, LocalSearch local_search, double alpha, int max_iter, int tabu_max_iter = 0);

    py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) override;

private:
    std::unique_ptr<WalshSurrogateLocalSearch> _local_search;
    double _alpha;
    int _max_iter;
};

class DRILSPlus : public ILS {
public:
    DRILSPlus(const std::vector<std::vector<int>>& masks, int N, DRILS::LocalSearch local_search, double alpha, double beta, int max_iter, int tabu_max_iter = 0);

    py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) override;
    
private:
    std::unique_ptr<WalshSurrogateLocalSearch> _local_search;
    double _alpha;
    double _beta;
    int _max_iter;
};
