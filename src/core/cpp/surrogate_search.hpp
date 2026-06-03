#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class SurrogateSearcher {
    public:
        SurrogateSearcher(const std::vector<std::vector<int>>& masks, int N);

        py::array_t<int> tabu_search(py::array_t<int> start_sol, py::array_t<double> coefs, int max_iter, unsigned int seed);

        py::array_t<int> hill_climber(py::array_t<int> start_sol, py::array_t<double> coefs);

    private:
        int _N;
        std::vector<std::vector<int>> _masks;
        std::vector<std::vector<int>> _bit_to_masks;  // précompilé une seule fois d'ou l'intêret de faire une classe, permet de retrouver les masques affecté par un changement de bits
};