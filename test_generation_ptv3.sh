#!/bin/bash

#SBATCH --job-name=test_incontext_ptv3_chair_epoch49 # name of job
#SBATCH --output=test_incontext_ptv3_chair_epoch49_%j.out # output file (%j = job ID)
#SBATCH --constraint=a100 # reserve 80 GB A100 GPUs
#SBATCH --nodes=1 # reserve 2 nodez
#SBATCH --ntasks=1 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:1 # reserve 8 GPUs per node
#SBATCH --cpus-per-task=8 # reserve 8 CPUs per task (and associated memory)
#SBATCH --time=05:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=mdv@a100 # A100 accounting

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default

module load cpuarch/amd # select modules compiled for AMD
module load pytorch-gpu/py3/2.1.1 # load modules

set -x # activate echo of launched commands
python test_generation_ptv3.py --category chair --dataroot /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/data/ShapeNetCore.v2.PC15k/ --model /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/output/train_generation_ptv3/2024-08-13-12-43-55/epoch_49.pth # execute script