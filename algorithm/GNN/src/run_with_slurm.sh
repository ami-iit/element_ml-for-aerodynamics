#!/bin/bash

#SBATCH --job-name=ironcub_aero_gnn       # Name of the job
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=4                 # CPUs per task
#SBATCH --mem=80G                         # Memory per node or per task
#SBATCH --time=23:30:00                   # Max runtime hh:mm:ss
#SBATCH --partition=gpua                  # Partition name
#SBATCH --gres=gpu:1                      # GPUs requested

# Run the training command
conda run -n ml-env python /fastwork/apaolino/element_ml-for-aerodynamics/algorithm/GNN/src/main.py /fastwork/apaolino/element_ml-for-aerodynamics/algorithm/GNN/test_cases/ironcub/input.cfg
