#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH -o log/%j-train_baseline.log
#SBATCH -c 10
#SBATCH --gres=gpu:volta:1

source /Users/wendysun/Desktop/curriculum-learning/stats/bin/activate

for dataset_type in 'foreground' 'composite' 'mix'
do
    python train_baseline.py \
    --epochs 10 \
    --dataset_type $dataset_type \
    --plot_name baseline_$dataset_type
done
