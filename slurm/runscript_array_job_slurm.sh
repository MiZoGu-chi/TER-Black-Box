#!/bin/bash

#SBATCH --job-name=run_test_surrogate  
#SBATCH --error=logs/%x_%j.err  
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=13-00:00:00

# Environement to load
if [ "$NEED_MODULE" == "1" ]; then
    module load python/3.11.4
fi

source env/bin/activate

# évidemment à modifier
python3 main.py "$@"