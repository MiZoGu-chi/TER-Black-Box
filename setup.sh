#!/bin/bash

CLUSTER_NAME=$(hostname)

if [[ "$CLUSTER_NAME" == *"nautilus"* ]]; then
    module load python/3.11.4
fi

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
