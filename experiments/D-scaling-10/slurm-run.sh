#!/bin/bash
#SBATCH --job-name=D-scaling-10
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=2
#SBATCH --time=0-5:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/D-scaling-10/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/D-scaling-10/logs/slurm-%A_%a.err
#SBATCH --mem=30GB
#SBATCH --array=0-1499

python /om2/user/ericjm/the-everything-machine/experiments/D-scaling-10/eval.py $SLURM_ARRAY_TASK_ID

