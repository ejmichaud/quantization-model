#!/bin/bash
#SBATCH --job-name=P-scaling-uniform-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-12:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/P-scaling-uniform-0/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/P-scaling-uniform-0/logs/slurm-%A_%a.err
#SBATCH --mem=15GB
#SBATCH --array=0-59

python /om2/user/ericjm/the-everything-machine/experiments/P-scaling-uniform-0/eval.py $SLURM_ARRAY_TASK_ID

