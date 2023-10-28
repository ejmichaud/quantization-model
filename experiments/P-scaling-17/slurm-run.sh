#!/bin/bash
#SBATCH --job-name=P-scaling-17
#SBATCH --gres=gpu:1
#SBATCH --ntasks=2
#SBATCH --time=0-12:00:00
#SBATCH --output=/om/user/ericjm/results/the-everything-machine/P-scaling-17/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/the-everything-machine/P-scaling-17/logs/slurm-%A_%a.err
#SBATCH --mem=15GB
#SBATCH --array=8,72,73,74,77,79,81,95,106,107,122,125,129,133,134,135,136,137,138,139,140,141,142,143,174,175,177,178,194,239,249,250,341,343,345,353,382,387,395,409,412,447,448,466

python /om2/user/ericjm/the-everything-machine/experiments/P-scaling-17/eval.py $SLURM_ARRAY_TASK_ID

