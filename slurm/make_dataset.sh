#!/bin/bash
#SBATCH --job-name=make_dataset
#SBATCH -o log/%j-make_dataset.log
#SBATCH -c 10
#SBATCH --gres=gpu:1

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH
export DATASET_DIR=/mnt/xfs/home/wendysun/curriculum_learning

sleep 0.1

echo "Building dataset..."

python preprocess_data.py
