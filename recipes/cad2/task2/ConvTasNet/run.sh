#!/bin/bash

#$ -V -cwd
#$ -l coproc_v100=1
#$ -l h_rt=48:00:00
#$ -l h_vmem=32G
#$ -pe smp 10
#$ -m be
#$ -j y
#$ -N convtasnet_train

set -e  # Exit on error
set -o pipefail

module purge
module load cuda

# Verify CUDA installation
nvcc --version
nvidia-smi

# Activate Conda environment
source ~/.bashrc
conda activate clarity

# Define the path to train.py
TRAIN_SCRIPT="/home/home01/sc22ol/Project/clarity/recipes/cad2/task2/ConvTasNet/train.py"

# Move to nobackup directory to avoid quota issues
cd /nobackup/sc22ol/

# Run training
CUDA_VISIBLE_DEVICES=0 python $TRAIN_SCRIPT \
    --exp_dir /nobackup/sc22ol/enhanced \
    --batch_size 4 \
    --aggregate 1 \
    --lr 0.0005 \
    --root /nobackup/sc22ol/metadata \
    --sample_rate 44100 \
    --segment 4.0 \
    --samples_per_track 64 \
    --causal True \
    --norm_type cLN \
    --music_tracks_file /nobackup/sc22ol/metadata

echo "Training completed."
