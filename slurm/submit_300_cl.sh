#!/bin/bash
#SBATCH --job-name=train_cl_300
#SBATCH -o log/%j-train_cl_300.log
#SBATCH -c 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=36:00:00

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1

module load conda
source activate cl_new

python train.py \
--epochs 300 \
--dataset_first 'foreground' \
--plot_name 'cl_300'
