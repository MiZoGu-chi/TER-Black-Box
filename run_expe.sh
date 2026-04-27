#!/bin/bash

for N in 64 128 256; do
    for K in 1 2 4 8; do
        sbatch --job-name=exp_N${N}_K${K}_tabu runscript_array_job_slurm.sh $N $K tabu
    done
done