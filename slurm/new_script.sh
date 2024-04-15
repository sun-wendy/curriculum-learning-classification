#!/bin/bash
#SBATCH --job-name=train_cl_600_18
#SBATCH -o log/%j-train_cl_600_18.log
#SBATCH -c 32
#SBATCH --gres=gpu:2080_ti:8
#SBATCH --time=800:00:00

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1

module load conda
source activate cl_new

python train.py \
--num_layers 18 \
--epochs 600 \
--dataset_first 'foreground' \
--plot_name 'cl_600_18'
