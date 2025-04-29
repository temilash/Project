#!/bin/bash
#SBATCH --job-name=cadenza-eval          # Job name
#SBATCH --time=48:00:00                  # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu                  # Request GPU partition
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=80G
#SBATCH --array=1-8                      # One task per instrument

# Load necessary modules
module load miniforge
conda activate clarity

# Array of instruments (ensure order matches index 1-8)
instruments=(Bassoon Cello Clarinet Flute Oboe Sax Viola Violin)

# Get the instrument for this job
instrument=${instruments[$SLURM_ARRAY_TASK_ID-1]}

echo "Evaluating model for instrument: $instrument"

# Define experiment folder
exp_dir="crm50/${instrument}"

# Run evaluation
python eval.py \
  --exp_dir "$exp_dir" \
  --out_dir out \
  --use_gpu 1
