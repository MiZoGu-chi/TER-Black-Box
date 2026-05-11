#!/bin/bash

#SBATCH --job-name=analyze_results
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --time=00:30:00        
#SBATCH --mem=8G               
#SBATCH --cpus-per-task=1


# Chargement de l'environnement
module load python/3.11.4
source env/bin/activate

python3 analyze.py "$1"