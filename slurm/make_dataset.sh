#!/bin/bash
#SBATCH --job-name=make_dataset
#SBATCH -o log/%j-make_dataset.log
#SBATCH -c 10
#SBATCH --gres=gpu:volta:1

export DATASET_DIR=/Users/wendysun/Desktop/curriculum-learning

source /Users/wendysun/Desktop/curriculum-learning/stats/bin/activate

python preprocess_data.py
