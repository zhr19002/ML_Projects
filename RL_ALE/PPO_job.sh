#!/bin/bash
#SBATCH -J PPO_job
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -o PPO_output.log
#SBATCH -e PPO_error.log

source /home/zhr19002/anaconda3/etc/profile.d/conda.sh
conda activate torch-sb3
cd /home/zhr19002/ML_projects/RL_ALE/
srun python PPO_train.py
