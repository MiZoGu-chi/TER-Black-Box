#!/bin/bash

#SBATCH --array=0-9
#SBATCH --error=slurm-err-%j.out
#SBATCH --output=slurm-o-%j.out	
#SBATCH --time=1:00:00
#SBATCH --ntasks=1             # How many tasks per node
#SBATCH --cpus-per-task=1       # Number of CPUs per task
#SBATCH --mem-per-cpu=10g       # Memory per CPU
#SBATCH --qos=short             # priority/quality of service

# Command to run
python main.py $1 $2 $3 $4 --seed $SLURM_ARRAY_TASK_ID 
