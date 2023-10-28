#!/bin/bash
#SBATCH --job-name=pythia-6
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --time=0-03:00:00
#SBATCH --output=/om/user/ericjm/results/phase-changes/pythia-6/logs/slurm-%A_%a.out
#SBATCH --mem=16GB
#SBATCH --array=9-9

python /om2/user/ericjm/phase-changes/experiments/pythia-6/eval.py $SLURM_ARRAY_TASK_ID

