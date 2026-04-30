#!/bin/bash

#SBATCH --job-name=run_test_tabu   
#SBATCH --error=logs/%x_%j.err  
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --qos=short             # priority/quality of service
#SBATCH --exclusive             # On resérve le noeud complet    

# Environement to load
module load python/3.11.4
source env/bin/activate

# évidemment à modifier
python main.py optimize
