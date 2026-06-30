#pragma once

#include <memory>
#include "walsh_surrogate_algorithm.hpp"
#include "local_search.hpp"

class DRILSBase : public WalshSurrogateAlgorithm {
public:
    enum class LocalSearch { 
        HC_BEST_IMPROVEMENT, 
        HC_FIRST_IMPROVEMENT, 
        TABU 
    };

    DRILSBase(const std::vector<std::vector<int>>& masks, int N, LocalSearch local_search, double alpha, int max_iter, int tabu_max_iter = 0);

    virtual ~DRILSBase() = default;

    virtual py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) = 0;

    std::vector<int> perturb(const std::vector<int>& solution, double alpha, unsigned int seed);

    std::vector<std::vector<int>> find_connected_components(const std::vector<bool>& is_differing, const py::array_t<double>& coefs);

    bool compare_component(const std::vector<int>& component, const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs);

    std::vector<int> partition_crossover(const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs);

protected:
    std::unique_ptr<WalshSurrogateLocalSearch> _local_search;
    double _alpha;
    int _max_iter;
};

class DRILS : public DRILSBase {
public:
    using DRILSBase::DRILSBase;

    py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) override;
};

class DRILSPlus : public DRILSBase {
public:
    DRILSPlus(const std::vector<std::vector<int>>& masks, int N, DRILSBase::LocalSearch local_search, double alpha, double beta, int max_iter, int tabu_max_iter = 0);

    py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) override;

protected:
    double _beta;
};

class DRILSPlusPlus : public DRILSBase {
public:
    DRILSPlusPlus(const std::vector<std::vector<int>>& masks, int N, DRILSBase::LocalSearch local_search, double alpha, double beta, double gamma, int max_iter, int tabu_max_iter = 0);

    py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) override;

protected:
    double _beta;
    double _gamma;
};
