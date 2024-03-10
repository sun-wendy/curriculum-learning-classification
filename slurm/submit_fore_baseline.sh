#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH -o log/%j-train_baseline.log
#SBATCH -c 32
#SBATCH --gres=gpu:v100:8
#SBATCH --time=200:00:00

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1

module load conda
source activate cl_new

python train_baseline.py \
--num_layers 34 \
--epochs 200 \
--dataset_type 'composite' \
--plot_name '34_baseline_composite'
