#!/bin/bash
#SBATCH -J LSTM_job
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -o LSTM_output.log
#SBATCH -e LSTM_error.log

source /home/zhr19002/anaconda3/etc/profile.d/conda.sh
conda activate tf-gpu
cd /home/zhr19002/ML_projects/DL_LSTM/
srun python hp_tuning_reg0.py