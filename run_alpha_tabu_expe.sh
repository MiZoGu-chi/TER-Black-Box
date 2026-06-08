#!/bin/bash

ALPHAS=(0.01 0.001 0.0001)
TABU_FACTORS=(2 10 50 100)
N=(64)
K=(1 2 4 8)

for alpha in "${ALPHAS[@]}"; do
    for factor in "${TABU_FACTORS[@]}"; do
        for n in "${N[@]}"; do
            for k in "${K[@]}"; do
                bash run_expe.sh optimize --N "$n" --K "$k" --budget 10000 --surrogate lasso_lars --algo cpp_tabu --alpha "$alpha" --tabu_factor "$factor" --duplicate_strategy random
            done
        done
    done
done
