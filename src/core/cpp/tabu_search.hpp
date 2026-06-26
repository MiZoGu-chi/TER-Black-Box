#pragma once

#include "local_search.hpp"

class TabuSearch : public WalshSurrogateLocalSearch {
public:
    TabuSearch(const std::vector<std::vector<int>>& masks, int N, int max_iter)
        : WalshSurrogateLocalSearch(masks, N), _max_iter(max_iter) {}

    TabuSearch(const std::vector<std::vector<int>>& masks, int N, const std::vector<std::vector<int>>& bit_to_masks, int max_iter)
        : WalshSurrogateLocalSearch(masks, N, bit_to_masks), _max_iter(max_iter) {}

    py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) override;

private:
    int _max_iter;
};
