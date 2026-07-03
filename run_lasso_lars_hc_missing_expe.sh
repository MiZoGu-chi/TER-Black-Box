#!/bin/bash

export CPUS_PER_TASK=20
export SLURM_QOS=quick
export SLURM_TIME=03:00:00

ALPHA=0.00001
N=64
K_LIST=(1 2 4 8)
D=2

declare -A MISSING

MISSING[hc_fi,1]="6:8 7:2"
MISSING[hc_fi,2]="4:4 4:6 5:0 5:4 6:0 6:7 6:9 7:0 7:1 7:2 7:3 7:4 7:5 7:6 7:7 7:9 8:1 8:2 8:3 8:4 8:5 8:6 8:7 8:8 8:9 9:1 9:3 9:4 9:5 9:6 9:7 9:8 9:9"
MISSING[hc_fi,4]="0:0 0:1 0:2 0:3 0:4 0:7 1:0 1:4 1:5 1:6 1:7 1:9 2:1 2:5 3:1 3:4 4:1 4:2 4:4 4:5 4:8 5:1 5:4 5:6 5:7 5:8 6:4 6:6 6:7 6:9 7:1 7:3 7:4 7:5 7:7 7:9 8:1 8:2 8:3 8:5 8:8 8:9 9:0 9:1 9:2 9:4 9:5 9:6 9:7"
MISSING[hc_fi,8]="0:1 0:2 0:3 0:5 0:6 0:7 0:8 0:9 1:1 1:2 1:3 1:5 1:7 1:8 2:1 2:2 2:3 2:4 2:5 2:6 3:1 3:2 3:8 4:1 4:2 4:4 4:5 4:7 4:9 5:0 5:1 5:4 5:5 5:8 6:2 6:9 7:2 7:8 8:5 8:7 8:8 9:1"

MISSING[hc_bi,1]=""
MISSING[hc_bi,2]="0:1 0:3 0:6 1:4 1:5 1:6 1:8 1:9 2:0 4:3 7:5 7:8 7:9 9:0 9:1 9:9"
MISSING[hc_bi,4]="0:0 2:0 2:7 2:8 3:3 3:8 4:0 4:8 4:9 5:0 6:3 6:5 6:6 6:7 6:8 6:9 7:1 7:3 7:4 7:5 7:6 7:7 7:8 8:0 8:5 8:6 8:7 8:8 9:1 9:4 9:5 9:6 9:7 9:9"
MISSING[hc_bi,8]="0:0 0:1 0:2 0:3 0:5 0:6 0:7 0:8 0:9 1:4 1:5 1:6 1:7 1:8 2:3 2:5 2:6 2:7 3:0 3:1 4:2 4:3 4:5 4:6 5:0 5:3 5:4 5:5 5:6 6:3 7:0 7:2 7:4 9:2 9:7"

for algo in hc_fi hc_bi; do
    for k in "${K_LIST[@]}"; do
        pairs="${MISSING[$algo,$k]}"
        for pair in $pairs; do
            instance="${pair%%:*}"
            seed="${pair##*:}"
            bash run_expe.sh optimize --N "$N" --K "$k" --D "$D" --budget 10000 \
                --surrogate lasso_lars --algo "$algo" --alpha "$ALPHA" \
                --duplicate_strategy random --instances "$instance" --seeds "$seed"
        done
    done
done
