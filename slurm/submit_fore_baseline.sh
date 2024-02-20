#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH -o log/%j-train_baseline.log
#SBATCH -c 28
#SBATCH --gres=gpu:v100:7
#SBATCH --time=96:00:00

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1

module load conda
source activate cl_new

python train_baseline.py \
--num_layers 18 \
--epochs 200 \
--dataset_type 'foreground' \
--plot_name $i'_baseline_foreground'
