#!/bin/bash

export CPUS_PER_TASK=1
export SLURM_QOS=short

ALPHAS=(0.00001)
TABU_FACTORS=(2)
N=(64)
K=(1)
D=(3)
N_SEEDS=10
N_INSTANCES=10

for alpha in "${ALPHAS[@]}"; do
    for factor in "${TABU_FACTORS[@]}"; do
        for n in "${N[@]}"; do
            for k in "${K[@]}"; do
                for instance in $(seq 0 $((N_INSTANCES - 1))); do
                    for seed in $(seq 0 $((N_SEEDS - 1))); do
                        bash run_expe.sh optimize --N "$n" --K "$k" --D "$D" --budget 10000 --surrogate lasso_lars --algo cpp_tabu --alpha "$alpha" --tabu_factor "$factor" --duplicate_strategy random --instances "$instance" --seeds "$seed"
                    done
                done
            done
        done
    done
done
