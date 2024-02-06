#!/bin/bash
#SBATCH --job-name=test_robustness
#SBATCH -o log/%j-test_robustness.log
#SBATCH -c 10
#SBATCH --gres=gpu:v100:1

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1

module load conda
source activate cl_new

python test_robustness.py
