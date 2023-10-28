#!/bin/bash
#SBATCH --job-name=pythia-4
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --time=0-03:00:00
#SBATCH --output=/om/user/ericjm/results/phase-changes/pythia-4/logs/slurm-%A_%a.out
#SBATCH --mem=16GB
#SBATCH --array=0-142

python /om2/user/ericjm/phase-changes/experiments/pythia-4/eval.py $SLURM_ARRAY_TASK_ID

