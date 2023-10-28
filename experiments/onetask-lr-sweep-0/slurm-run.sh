#!/bin/bash
#SBATCH --job-name=onetask-lr-sweep-0
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-2:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/onetask-lr-sweep-0/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/onetask-lr-sweep-0/logs/slurm-%A_%a.err
#SBATCH --mem=8G
#SBATCH --constraint=6GB
#SBATCH --array=0-15

python /om2/user/ericjm/the-everything-machine/experiments/onetask-lr-sweep-0/eval.py $SLURM_ARRAY_TASK_ID


