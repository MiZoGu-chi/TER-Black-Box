from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="surrogate_search",
    ext_modules=[
        Pybind11Extension(
            "src.core.cpp.surrogate_search",
            [
                "src/core/cpp/walsh_surrogate_algorithm.cpp",
                "src/core/cpp/local_search.cpp",
                "src/core/cpp/hill_climber.cpp",
                "src/core/cpp/tabu_search.cpp",
                "src/core/cpp/drils.cpp",
                "src/core/cpp/drils.cpp",
                "src/core/cpp/bindings.cpp",
            ],
            cxx_std=17,
            extra_compile_args=["-O3", "-march=native"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
