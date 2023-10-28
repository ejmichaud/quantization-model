#!/bin/bash
#SBATCH --job-name=pythia-2
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=2
#SBATCH --time=0-36:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/pythia-2/logs/slurm-%A_%a.out
#SBATCH --mem=60GB
#SBATCH --array=5-6

python /om2/user/ericjm/the-everything-machine/experiments/pythia-2/eval.py $SLURM_ARRAY_TASK_ID

