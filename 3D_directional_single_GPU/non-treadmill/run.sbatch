#!/bin/bash

# Use command "sbatch run.sbatch" to submit job
# This is an example for submitting sbatch job using slurm job manager

#SBATCH --job-name=bash
#SBATCH --output=output%j.out
#SBATCH --error=output%j.err
#SBATCH -N 1                        # number of node (1 GPU per node in Discovery).
#SBATCH -n 2                       # number of MPI process
#SBATCH -t 0-08:00:00
#SBATCH --mem=64Gb
##SBATCH --exclusive
#SBATCH --partition multigpu             # Submit job to the new partition
#SBATCH --gres=gpu:v100-pcie:1
##SBATCH -w, --nodelist=d1002
#SBATCH --exclude=d1005,c2190             # Exclude specific node

device=0

name=test0
nvcc -arch=sm_70 *.cu -o test0

./test0 $device >> out.txt