#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH -o log/%j-train_baseline.log
#SBATCH -c 10
#SBATCH --gres=gpu:v100:1

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1


python train_baseline.py \
--num_layers 50 \
--epochs 50 \
--dataset_type 'foreground' \
--plot_name 'baseline_foreground'
