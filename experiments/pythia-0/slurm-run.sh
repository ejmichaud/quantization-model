#!/bin/bash
#SBATCH --job-name=pythia-0
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=2
#SBATCH --time=0-01:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/pythia-0/logs/slurm-%A_%a.out
#SBATCH --mem=60GB
#SBATCH --array=7-7

python /om2/user/ericjm/the-everything-machine/experiments/pythia-0/eval.py $SLURM_ARRAY_TASK_ID

