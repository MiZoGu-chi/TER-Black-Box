#include "walsh_surrogate_algorithm.hpp"

WalshSurrogateAlgorithm::WalshSurrogateAlgorithm(const std::vector<std::vector<int>>& masks, int N)
    : _N(N), _masks(masks), _bit_to_masks(N) {
    for (int i = 0; i < (int)_masks.size(); i++)
        for (int b : _masks[i])
            _bit_to_masks[b].push_back(i);
}

WalshSurrogateAlgorithm::WalshSurrogateAlgorithm(const std::vector<std::vector<int>>& masks, int N, const std::vector<std::vector<int>>& bit_to_masks)
    : _N(N), _masks(masks), _bit_to_masks(bit_to_masks) {}

std::vector<int> WalshSurrogateAlgorithm::to_vector(const py::array_t<int>& a) {
    auto acc = a.unchecked<1>();
    std::vector<int> v(acc.shape(0));
    for (int i = 0; i < (int)v.size(); i++) v[i] = acc(i);
    return v;
}

py::array_t<int> WalshSurrogateAlgorithm::to_array(const std::vector<int>& v) {
    py::array_t<int> result(v.size());
    auto r = result.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); i++) r(i) = v[i];
    return result;
}

double WalshSurrogateAlgorithm::evaluate_fitness(const std::vector<int>& solution, const py::array_t<double>& coefs_) {
    auto coefs = coefs_.unchecked<1>();
    double fit = 0.0;

    for (int i = 0; i < (int)_masks.size(); i++) {
        if (coefs(i) == 0.0) continue;

        int s = 0;
        for (int b : _masks[i]) s += solution[b];

        fit += coefs(i) * ((s % 2 == 0) ? 1.0 : -1.0);
    }

    return fit;
}
