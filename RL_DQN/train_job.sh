#!/bin/bash
#SBATCH -J train_job
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -o output.log
#SBATCH -e error.log

source /home/zhr19002/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-sb3
cd /home/zhr19002/ML_projects/RL_DQN/
srun python train.py