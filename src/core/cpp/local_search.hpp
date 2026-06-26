#pragma once

#include "walsh_surrogate_algorithm.hpp"

class WalshSurrogateLocalSearch : public WalshSurrogateAlgorithm {

public:
    WalshSurrogateLocalSearch(const std::vector<std::vector<int>>& masks, int N)
        : WalshSurrogateAlgorithm(masks, N) {}

    WalshSurrogateLocalSearch(const std::vector<std::vector<int>>& masks, int N, const std::vector<std::vector<int>>& bit_to_masks)
        : WalshSurrogateAlgorithm(masks, N, bit_to_masks) {}

    virtual ~WalshSurrogateLocalSearch() = default;

    virtual py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) = 0;

protected:
    void init_walsh_state(const py::array_t<int>& sol, const py::array_t<double>& coefs,std::vector<int>& current, std::vector<int>& best, std::vector<double>& phi, std::vector<double>& delta, double& cur_fit);

    void apply_incremental_flip(int bit, const py::array_t<double>& coefs, std::vector<int>& current, std::vector<double>& phi, std::vector<double>& delta, double& cur_fit);
};
