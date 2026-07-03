#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// classe qui maintient l'état incrémental
class WalshSurrogateAlgorithm {

public:
    WalshSurrogateAlgorithm(const std::vector<std::vector<int>>& masks, int N);

    WalshSurrogateAlgorithm(const std::vector<std::vector<int>>& masks, int N, const std::vector<std::vector<int>>& bit_to_masks);

    virtual ~WalshSurrogateAlgorithm() = default;

    std::vector<int> to_vector(const py::array_t<int>& a);
    py::array_t<int> to_array(const std::vector<int>& v);

    double evaluate_fitness(const std::vector<int>& solution, const py::array_t<double>& coefs);

protected:
    int _N;
    std::vector<std::vector<int>> _masks;
    std::vector<std::vector<int>> _bit_to_masks; 
};
