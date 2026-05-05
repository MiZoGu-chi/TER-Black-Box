#!/bin/bash

#SBATCH --job-name=run_test_tabu   
#SBATCH --error=logs/%x_%j.err  
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --nodes=1
#SBATCH --partition=intel-E5-2695

# Environement to load
#module load python/3.11.4
source env/bin/activate

# évidemment à modifier
python3 main.py "$@"