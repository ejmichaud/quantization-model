#!/bin/bash
#SBATCH --job-name=D-scaling-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-24:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/D-scaling-0/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/D-scaling-0/logs/slurm-%A_%a.err
#SBATCH --mem=8G
#SBATCH --constraint=6GB
#SBATCH --array=0-29

python /om2/user/ericjm/the-everything-machine/experiments/D-scaling-0/D-scaling-0-config.py $SLURM_ARRAY_TASK_ID


