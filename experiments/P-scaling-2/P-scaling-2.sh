#!/bin/bash
#SBATCH --job-name=P-scaling-2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=0-08:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/P-scaling-2/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/P-scaling-2/logs/slurm-%A_%a.err
#SBATCH --mem=4G
#SBATCH --array=0-19

source ~/.bash_profile
conda activate neural-scaling
python /om2/user/ericjm/the-everything-machine/experiments/P-scaling-2/P-scaling-2-config.py $SLURM_ARRAY_TASK_ID


