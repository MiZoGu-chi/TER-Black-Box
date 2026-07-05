#!/bin/bash

export CPUS_PER_TASK=20
export SLURM_QOS=quick
export SLURM_TIME=03:00:00

ALPHA=0.00001
N=64
K_LIST=(1 2 4 8)
D=2

declare -A MISSING

MISSING[hc_fi,1]=""
MISSING[hc_fi,2]=""
MISSING[hc_fi,4]="1:0 1:4 1:5 1:6 1:7 4:5"
MISSING[hc_fi,8]="2:3"

MISSING[hc_bi,1]=""
MISSING[hc_bi,2]=""
MISSING[hc_bi,4]="6:5 6:6 6:7 6:8 6:9 7:1 7:8 8:0 8:5 8:6 8:7 8:8 9:5 9:6"
MISSING[hc_bi,8]=""

for algo in hc_fi hc_bi; do
    for k in "${K_LIST[@]}"; do
        pairs="${MISSING[$algo,$k]}"
        for pair in $pairs; do
            instance="${pair%%:*}"
            seed="${pair##*:}"
            bash slurm/run_expe.sh optimize --N "$N" --K "$k" --D "$D" --budget 10000 \
                --surrogate lasso_lars --algo "$algo" --alpha "$ALPHA" \
                --duplicate_strategy random --instances "$instance" --seeds "$seed"
        done
    done
done
