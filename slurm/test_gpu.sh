#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH -o log/%j-test_gpu.log
#SBATCH -c 10
#SBATCH --gres=gpu:1

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

sleep 0.1

conda activate cl_env

python test_gpu.py
