#!/bin/bash
#SBATCH --job-name=D-scaling-4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-24:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/D-scaling-4/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/D-scaling-4/logs/slurm-%A_%a.err
#SBATCH --mem=12GB
#SBATCH --constraint=12GB
#SBATCH --array=0-29

python /om2/user/ericjm/the-everything-machine/experiments/D-scaling-4/eval.py $SLURM_ARRAY_TASK_ID

