#!/bin/bash
#SBATCH --job-name=both-scaling-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-12:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/both-scaling-0/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/both-scaling-0/logs/slurm-%A_%a.err
#SBATCH --mem=12GB
#SBATCH --array=0-168

python /om2/user/ericjm/the-everything-machine/experiments/both-scaling-0/eval.py $SLURM_ARRAY_TASK_ID

