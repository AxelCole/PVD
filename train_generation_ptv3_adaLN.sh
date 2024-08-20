#!/bin/bash

# Define job name

#SBATCH --job-name=train_adaLN_ptv3_2xA100 # Use the variable for the job name
#SBATCH --output=train_adaLN_ptv3_2xA100_%j.out # Use the variable in the output file name
#SBATCH --constraint=a100 # reserve 80 GB A100 GPUs
#SBATCH --nodes=1 # reserve 1 node
#SBATCH --ntasks=2 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:2 # reserve 8 GPUs per node
#SBATCH --cpus-per-task=8 # reserve 8 CPUs per task (and associated memory)
#SBATCH --time=20:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=mdv@a100 # A100 accounting

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default

module load cpuarch/amd # select modules compiled for AMD
module load pytorch-gpu/py3/2.1.1 # load modules

set -x # activate echo of launched commands
python train_generation_ptv3_adaLN.py --category chair --bs 16 --dataroot /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/data/ShapeNetCore.v2.PC15k/ --distribution_type multi # execute script