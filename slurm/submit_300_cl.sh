#!/bin/bash
#SBATCH --job-name=train_cl_600
#SBATCH -o log/%j-train_cl_600.log
#SBATCH -c 28
#SBATCH --gres=gpu:v100:7
#SBATCH --time=500:00:00

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1

module load conda
source activate cl_new

python train.py \
--num_layers 50 \
--epochs 600 \
--dataset_first 'foreground' \
--plot_name 'cl_600'
