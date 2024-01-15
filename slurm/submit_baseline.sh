#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH -o log/%j-train_baseline.log
#SBATCH -c 10
#SBATCH --gres=gpu:volta:1

source source /mnt/xfs/home/wendysun/curriculum_learning/stats/bin/activate

for dataset_type in 'foreground' 'composite' 'mix'
do
    python curriculum_learning/train_baseline.py \
    --epochs 10 \
    --dataset_type $dataset_type \
    --plot_name baseline_$dataset_type
done
