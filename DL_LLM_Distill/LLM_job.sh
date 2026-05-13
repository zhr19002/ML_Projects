#!/bin/bash
#SBATCH -J LLM_job
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -o LLM_output.log
#SBATCH -e LLM_error.log

source /home/zhr19002/anaconda3/etc/profile.d/conda.sh
conda activate chronos
cd /home/zhr19002/ML_Projects/DL_LLM_Distill/
srun python student_module.py