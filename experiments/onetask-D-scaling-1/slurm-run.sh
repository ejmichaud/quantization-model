#!/bin/bash
#SBATCH --job-name=onetask-D-scaling-1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-1:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/onetask-D-scaling-1/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/onetask-D-scaling-1/logs/slurm-%A_%a.err
#SBATCH --mem=8G
#SBATCH --constraint=6GB
#SBATCH --array=0-18

python /om2/user/ericjm/the-everything-machine/experiments/onetask-D-scaling-1/eval.py $SLURM_ARRAY_TASK_ID


