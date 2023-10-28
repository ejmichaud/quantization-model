#!/bin/bash
#SBATCH --job-name=P-scaling-5
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-24:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/P-scaling-5/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/P-scaling-5/logs/slurm-%A_%a.err
#SBATCH --mem=8G
#SBATCH --constraint=6GB
#SBATCH --array=0-59

python /om2/user/ericjm/the-everything-machine/experiments/P-scaling-5/P-scaling-5-config.py $SLURM_ARRAY_TASK_ID

