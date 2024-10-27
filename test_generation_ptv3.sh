#!/bin/bash

#SBATCH --job-name=test_adaLN_ptv3_grad_clip1_no_flash_attn_drop_010_grid_size_010_patch_size_256_no_clipped_chair_epoch_1249 # name of job
#SBATCH --output=test_adaLN_ptv3_grad_clip1_no_flash_attn_drop_010_grid_size_010_patch_size_256_no_clipped_chair_epoch_1249_%j.out # output file (%j = job ID)
#SBATCH --constraint=a100 # reserve 80 GB A100 GPUs
#SBATCH --nodes=1 # reserve 2 nodes
#SBATCH --ntasks=1 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:1 # reserve 8 GPUs per node
#SBATCH --cpus-per-task=8 # reserve 8 CPUs per task (and associated memory)
#SBATCH --time=06:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=mdv@a100 # A100 accounting

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default

module load arch/a100 # select modules compiled for AMD
module load pytorch-gpu/py3/2.1.1 # load modules 

set -x # activate echo of launched commands
python test_generation_ptv3.py --category chair --dataroot /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/data/ShapeNetCore.v2.PC15k/ --model /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/output/train_generation_ptv3_adaLN/2024-10-04-10-49-00/epoch_1249.pth --grid_size 0.1 --enc_patch_size '(256, 256, 256, 256, 256)' --dec_patch_size '(256, 256, 256, 256)' # execute script