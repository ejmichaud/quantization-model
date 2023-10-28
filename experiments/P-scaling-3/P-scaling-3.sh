#!/bin/bash
#SBATCH --job-name=P-scaling-3
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00
#SBATCH --output=/om2/user/ericjm/the-everything-machine/results/P-scaling-3/logs/slurm-%A_%a.out
#SBATCH --error=/om2/user/ericjm/the-everything-machine/results/P-scaling-3/logs/slurm-%A_%a.err
#SBATCH --mem=8G
#SBATCH --constraint=20GB
#SBATCH --array=0-19

source ~/.bash_profile
conda activate neural-scaling
python /om2/user/ericjm/the-everything-machine/experiments/P-scaling-3/P-scaling-3-config.py $SLURM_ARRAY_TASK_ID


