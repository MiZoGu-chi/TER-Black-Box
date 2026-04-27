#!/bin/bash

#SBATCH --array=0-9
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --output=logs/%x_%A_%a.out	
#SBATCH --time=1:00:00
#SBATCH --ntasks=1             # How many tasks per node
#SBATCH --cpus-per-task=1       # Number of CPUs per task
#SBATCH --mem-per-cpu=10g       # Memory per CPU
#SBATCH --qos=short             # priority/quality of service

# Environement to load
module load python/3.11.4
source env/bin/activate

# Command to run
python main.py $1 $2 --seed $SLURM_ARRAY_TASK_ID optimize --algo $3 
