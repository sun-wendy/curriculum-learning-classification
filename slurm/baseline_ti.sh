#!/bin/bash
#SBATCH --job-name=baseline_ti
#SBATCH -o log/%j-baseline_ti.log
#SBATCH -c 32
#SBATCH --gres=gpu:2080_ti:8
#SBATCH --time=200:00:00

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

export OMP_NUM_THREADS=1
export USE_NNPACK=0

sleep 0.1

module load conda
source activate cl_new

python train_on_ti_test.py \
--num_layers 50 \
--epochs 200 \
--dataset_type 'mix' \
--plot_name '50_baseline_mix'
