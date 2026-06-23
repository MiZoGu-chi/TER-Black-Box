#!/bin/bash

ALPHAS=(0.00001)
DRILS_FACTORS=(2)
N=(64)
K=(1)
D=(2)

for alpha in "${ALPHAS[@]}"; do
    for factor in "${DRILS_FACTORS[@]}"; do
        for n in "${N[@]}"; do
            for k in "${K[@]}"; do
                bash run_expe.sh optimize --N "$n" --K "$k" --D "$D" --budget 10000 --surrogate lasso_lars --algo cpp_drils_hc_fi --alpha "$alpha" --drils_factor "$factor" --duplicate_strategy random --instances 0
            done
        done
    done
done
