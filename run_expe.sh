#!/bin/bash

CLUSTER_NAME=$(hostname)

if [[ "$CLUSTER_NAME" == *"nautilus"* ]]; then
    export NEED_MODULE=1
    sbatch \
        --qos=short \
        --cpus-per-task=20 \
        --mem-per-cpu=4G \
        --nodes=1 \
        slurm/runscript_array_job_slurm.sh "$@"

elif [[ "$CLUSTER_NAME" == "stargate" ]]; then
    export NEED_MODULE=0
    sbatch \
        --partition=ram \
        --cpus-per-task=40 \
        --mem-per-cpu=10G \
        --nodes=1 \
        slurm/runscript_array_job_slurm.sh "$@"
fi