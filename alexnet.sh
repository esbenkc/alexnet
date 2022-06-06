#!/bin/bash
#SBATCH --job-name=exam
#SBATCH --output=exam/nn.out
#SBATCH --error=exam/nn.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=eskchris@ucsc.edu
#SBATCH --partition=gpuq
#SBATCH --account=gpuq

srun ./alexnet.out
# srun ./a.out