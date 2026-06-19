#!/bin/bash

ALPHAS=(0.0001 0.00001)
TABU_FACTORS=(2)
N=(64)
K=(1 2 4 8)
D=(3)

for alpha in "${ALPHAS[@]}"; do
    for factor in "${TABU_FACTORS[@]}"; do
        for n in "${N[@]}"; do
            for k in "${K[@]}"; do
                bash run_expe.sh optimize --N "$n" --K "$k" --D "$D" --budget 10000 --surrogate lasso_lars --algo cpp_tabu --alpha "$alpha" --tabu_factor "$factor" --duplicate_strategy random
            done
        done
    done
done
