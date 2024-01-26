#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH -o log/%j-train_baseline.log
#SBATCH -c 10
#SBATCH --gres=gpu:1

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

sleep 0.1

module load conda
conda activate cl_env

python test_gpu.py
