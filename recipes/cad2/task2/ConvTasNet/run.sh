#!/bin/bash
#SBATCH --job-name=cadenza-job          # Job name
#SBATCH --time=48:00:00                  # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu                 # Request GPU partition
#SBATCH --gres=gpu:1                # Request 2 GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=80G
#SBATCH --array=1-8                     # One task per instrument (adjust if needed)

# Load necessary modules
module load miniforge
conda activate clarity

# Define an array of instruments (order matters; ensure indices match your --array range)
instruments=(Bassoon Cello Clarinet Flute Oboe Sax Viola Violin)

# SLURM_ARRAY_TASK_ID is 1-indexed, so subtract 1 for zero-indexed bash arrays
instrument=${instruments[$SLURM_ARRAY_TASK_ID-1]}

echo "Running experiment for instrument: ${instrument}"

# Run train.py with configuration overrides
python train.py \
  --exp_dir "crm50/${instrument}" \
  --target "$instrument" \
  --lr 0.0005 \
