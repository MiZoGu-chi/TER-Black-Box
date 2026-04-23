#!/bin/bash

for N in 64 128 256; do
for K in 1 2 4 8; do

sbatch runscript_array_job_slurm.sh $N $K Tabu 0.3

done
done
