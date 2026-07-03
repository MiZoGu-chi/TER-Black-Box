#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "hill_climber.hpp"
#include "tabu_search.hpp"
#include "drils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(surrogate_search, m) {
    auto hc = py::class_<HillClimber>(m, "HillClimber")
        .def(py::init([](const std::vector<std::vector<int>>& masks, int N, HillClimber::Strategy strat) {
                 return HillClimber(masks, N, HillClimber::make_strategy(strat));
            }),
            py::arg("masks"), 
            py::arg("N"), 
            py::arg("strat"))
        .def("search", &HillClimber::search,
            py::arg("start_sol"), 
            py::arg("coefs"), 
            py::arg("seed"));

    py::enum_<HillClimber::Strategy>(hc, "Strategy")
        .value("BEST_IMPROVEMENT", HillClimber::Strategy::BEST_IMPROVEMENT)
        .value("FIRST_IMPROVEMENT", HillClimber::Strategy::FIRST_IMPROVEMENT);

    py::class_<TabuSearch>(m, "TabuSearch")
        .def(py::init<const std::vector<std::vector<int>>&, int, int>(),
            py::arg("masks"), 
            py::arg("N"), 
            py::arg("max_iter"))
        .def("search", &TabuSearch::search,
            py::arg("start_sol"), 
            py::arg("coefs"), 
            py::arg("seed"));

    auto drils = py::class_<DRILS>(m, "DRILS")
        .def(py::init<const std::vector<std::vector<int>>&, int, DRILS::LocalSearch, double, int, int>(),
            py::arg("masks"), 
            py::arg("N"), 
            py::arg("local_search"),
            py::arg("alpha"), 
            py::arg("drils_max_iter"), 
            py::arg("tabu_max_iter") = 0)
        .def("search", &DRILS::search,
            py::arg("start_sol"),
            py::arg("coefs"),
            py::arg("seed"));

    py::enum_<DRILS::LocalSearch>(drils, "LocalSearch")
        .value("HC_BEST_IMPROVEMENT", DRILS::LocalSearch::HC_BEST_IMPROVEMENT)
        .value("HC_FIRST_IMPROVEMENT", DRILS::LocalSearch::HC_FIRST_IMPROVEMENT)
        .value("TABU", DRILS::LocalSearch::TABU);

    auto drils_plus = py::class_<DRILSPlus>(m, "DRILSPlus")
        .def(py::init<const std::vector<std::vector<int>>&, int, DRILSBase::LocalSearch, double, double, int, int>(),
            py::arg("masks"),
            py::arg("N"),
            py::arg("local_search"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("drils_max_iter"),
            py::arg("tabu_max_iter") = 0)
        .def("search", &DRILSPlus::search,
            py::arg("start_sol"),
            py::arg("coefs"),
            py::arg("seed"));

    auto drils_plus_plus = py::class_<DRILSPlusPlus>(m, "DRILSPlusPlus")
        .def(py::init<const std::vector<std::vector<int>>&, int, DRILSBase::LocalSearch, double, double, double, int, int>(),
            py::arg("masks"),
            py::arg("N"),
            py::arg("local_search"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("gamma"),
            py::arg("drils_max_iter"),
            py::arg("tabu_max_iter") = 0)
        .def("search", &DRILSPlusPlus::search,
            py::arg("start_sol"),
            py::arg("coefs"),
            py::arg("seed"));
}
