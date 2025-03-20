#!/bin/bash
#SBATCH --job-name=cadenza-job          # Job name
#SBATCH --time=18:00:00            # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu            # Request GPU partition
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=80G

# Load any necessary modules
module load miniforge

# Activate Conda environment
conda activate clarity

set -e  # Exit if any command fails
set -o pipefail  # Catch errors in pipelines

# Run the scripts
python train.py
python eval.py

echo "Both scripts ran successfully!"
