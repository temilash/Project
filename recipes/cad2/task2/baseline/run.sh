#!/bin/bash
#SBATCH --job-name=ml-job          # Job name
#SBATCH --time=09:00:00            # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu            # Request GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=80G     

# Load any necessary modules
module load miniforge

# Activate Conda environment
conda activate clarity

# Run the scripts
python enhance.py
python evaluate.py

echo "Both scripts ran successfully!"
