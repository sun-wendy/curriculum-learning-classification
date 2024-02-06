#!/bin/bash
#SBATCH --job-name=augment_dataset
#SBATCH -o log/%j-augment_dataset.log
#SBATCH -c 10
#SBATCH --gres=gpu:1

export PATH=/mnt/xfs/home/wendysun/curriculum_learning:$PATH

sleep 0.1

module load conda
source activate cl_new

echo "Augmenting dataset..."

python augment_data.py
