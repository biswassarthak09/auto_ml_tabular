#!/bin/bash
#SBATCH --job-name=od3d_train_no_sphere
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00


# Activate conda environment
conda init
conda activate automl_env

# Run training script
python src/automl_pipeline/nas_hpo_optuna.py