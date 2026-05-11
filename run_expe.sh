#!/bin/bash

CLUSTER_NAME=$(hostname)

if [[ "$CLUSTER_NAME" == *"nautilus"* ]]; then
    export NEED_MODULE=1
    sbatch \
        --qos=short \
        --cpus-per-task=32 \
        --mem-per-cpu=4G \
        --nodes=1 \
        --time=05:00:00 \
        runscript_array_job_slurm.sh "$@"

elif [[ "$CLUSTER_NAME" == "stargate" ]]; then
    export NEED_MODULE=0
    sbatch \
        --partition=intel-E5-2695 \
        --exclusive \
        --nodes=1 \
        --time=05:00:00 \
        runscript_array_job_slurm.sh "$@"
fi