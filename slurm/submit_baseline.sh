#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH -o log/%j-train_baseline.log
#SBATCH -c 10
#SBATCH --gres=gpu:1

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1

for dataset_type in 'foreground' 'composite' 'mix'
do
    python train_baseline.py \
    --epochs 1 \
    --dataset_type $dataset_type \
    --plot_name baseline_$dataset_type
done
