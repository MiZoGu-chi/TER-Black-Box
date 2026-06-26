#pragma once

#include <memory>
#include <random>
#include "local_search.hpp"

class HillClimberStrategy {
public:
    virtual ~HillClimberStrategy() = default;
    virtual int select(const std::vector<double>& delta, std::mt19937& rng) = 0;
};

class HillClimberStrategyBI : public HillClimberStrategy {
public:
    int select(const std::vector<double>& delta, std::mt19937& rng) override;
};

class HillClimberStrategyFI : public HillClimberStrategy {
public:
    int select(const std::vector<double>& delta, std::mt19937& rng) override;
};

class HillClimber : public WalshSurrogateLocalSearch {
public:
    enum class Strategy { BEST_IMPROVEMENT, FIRST_IMPROVEMENT };

    static std::unique_ptr<HillClimberStrategy> make_strategy(Strategy s);

    HillClimber(const std::vector<std::vector<int>>& masks, int N, std::unique_ptr<HillClimberStrategy> strategy)
        : WalshSurrogateLocalSearch(masks, N), _strategy(std::move(strategy)) {}

    HillClimber(const std::vector<std::vector<int>>& masks, int N, const std::vector<std::vector<int>>& bit_to_masks, std::unique_ptr<HillClimberStrategy> strategy)
        : WalshSurrogateLocalSearch(masks, N, bit_to_masks), _strategy(std::move(strategy)) {}

    py::array_t<int> search(py::array_t<int> start_sol, py::array_t<double> coefs, unsigned int seed) override;

private:
    std::unique_ptr<HillClimberStrategy> _strategy;
};
