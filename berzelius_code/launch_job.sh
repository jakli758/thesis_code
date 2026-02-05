#!/bin/bash

#SBATCH --gpus 2
#SBATCH -t 3-00:00:00
#SBATCH -N 1

cp -r /proj/afraid/users/x_jakli/data/accepted_preprocessed_128 /scratch/local

cd /scratch/local/accepted_preprocessed_128

module load Miniforge3/24.7.1-2-hpc1-bdist

conda activate /proj/afraid/users/x_jakli/thesis312

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/afraid/users/x_jakli/thesis312/lib

torchrun --standalone --nproc_per_node=2 train_multigpu.py 50 10

cp -r /scratch/local/accepted_preprocessed_128/results/* /proj/afraid/users/x_jakli/accepted_preprocessed_128/results/test_run/