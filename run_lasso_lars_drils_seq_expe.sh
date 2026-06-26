#!/bin/bash

export CPUS_PER_TASK=1

ALPHAS=(0.00001)
DRILS_FACTORS=(2)
N=(64)
K=(1)
D=(2)
N_SEEDS=10

for alpha in "${ALPHAS[@]}"; do
    for factor in "${DRILS_FACTORS[@]}"; do
        for n in "${N[@]}"; do
            for k in "${K[@]}"; do
                for seed in $(seq 0 $((N_SEEDS - 1))); do
                    bash run_expe.sh optimize --N "$n" --K "$k" --D "$D" --budget 10000 --surrogate lasso_lars --algo cpp_drils_tabu --alpha "$alpha" --drils_factor "$factor" --duplicate_strategy random --instances 0 --seeds "$seed"
                done
            done
        done
    done
done
