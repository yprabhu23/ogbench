#!/bin/bash

# --- SLURM CONFIGURATION ---
SBATCH --job-name=visual_cube_gcbc    # Name of the job
SBATCH -p mit_normal_gpu              # Partition (GPU)
SBATCH --gres=gpu:h100:1              # Request 1 H100 GPU
SBATCH -t 06:00:00                    # Time limit (Max is 6h for this partition)
SBATCH -N 1                           # Request 1 node
SBATCH -c 16                          # Request 16 CPU cores (good for dataloading)
SBATCH --mem=64G                      # Request 64GB RAM (adjust if OOM occurs)
SBATCH -o logs/%x_%j.out              # Standard output log (make sure 'logs' dir exists)
SBATCH -e logs/%x_%j.err              # Standard error log

# --- ENVIRONMENT SETUP ---
# Initialize Conda (needed because .bashrc isn't always sourced in batch mode)
source ~/.bashrc

# Activate your environment
conda activate dlenv

# Debug: Print GPU status to log to verify H100 was acquired
echo "Job started on $(hostname) at $(date)"
echo "GPU Info:"
nvidia-smi

# --- RUN YOUR COMMAND ---
python main.py \
    --env_name=visual-cube-single-play-v0 \
    --train_steps=500000 \
    --eval_episodes=50 \
    --eval_on_cpu=0 \
    --agent=agents/gcbc.py \
    --agent.batch_size=256 \
    --agent.encoder=vit_small \
    --agent.p_aug=0.5

echo "Job finished at $(date)"