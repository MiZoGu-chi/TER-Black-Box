#!/bin/bash

#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --output=logs/%x_%A_%a.out	
#SBATCH --time=5:00:00
#SBATCH --ntasks=1              # How many tasks per node
#SBATCH --qos=short             # priority/quality of service
#SBATCH --exclusive             # On resérve le noeud complet    

# Environement to load
module load python/3.11.4
source env/bin/activate

# Command to run
python main.py $1 $2 optimize --algo $3 
