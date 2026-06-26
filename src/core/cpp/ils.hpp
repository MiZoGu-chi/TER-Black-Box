#pragma once

#include "walsh_surrogate_algorithm.hpp"

// Base partagée par DRILS, DRILSPlus et PRILS : perturbation et recombinaison
// par partition crossover (PX) sur les composantes connexes des bits actifs.
class ILS : public WalshSurrogateAlgorithm {

public:
    ILS(const std::vector<std::vector<int>>& masks, int N)
        : WalshSurrogateAlgorithm(masks, N) {}

    virtual ~ILS() = default;

    virtual py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) = 0;

protected:
    std::vector<int> perturb(const std::vector<int>& solution, double alpha, unsigned int seed);

    std::vector<std::vector<int>> find_connected_components(const std::vector<bool>& is_differing, const py::array_t<double>& coefs);

    bool compare_component(const std::vector<int>& component, const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs);

    std::vector<int> partition_crossover(const std::vector<int>& p1, const std::vector<int>& p2, py::array_t<double> coefs);
};
