#!/bin/bash

CLUSTER_NAME=$(hostname)

if [[ "$CLUSTER_NAME" == *"nautilus"* ]]; then
    export NEED_MODULE=1
    CPUS=${CPUS_PER_TASK:-20}
    sbatch \
        --qos=short \
        --cpus-per-task="$CPUS" \
        --mem-per-cpu=8G \
        --nodes=1 \
        slurm/job.sh "$@"

elif [[ "$CLUSTER_NAME" == "stargate" ]]; then
    export NEED_MODULE=0
    sbatch \
        --partition=ram \
        --cpus-per-task=40 \
        --mem-per-cpu=10G \
        --nodes=1 \
        slurm/job.sh "$@"
fi