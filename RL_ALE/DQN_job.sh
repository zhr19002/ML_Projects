#!/bin/bash
#SBATCH -J DQN_job
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -o DQN_output.log
#SBATCH -e DQN_error.log

source /home/zhr19002/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-sb3
cd /home/zhr19002/ML_projects/RL_ALE/
srun python DQN_train.py