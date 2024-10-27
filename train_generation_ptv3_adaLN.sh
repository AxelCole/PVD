#!/bin/bash

# Define job name

#SBATCH --job-name=train_adaLN_ptv3_grad_clip1_no_flash_attn_drop_010_grid_size_010_patch_size_256_2xA100 # Use the variable for the job name
#SBATCH --output=train_adaLN_ptv3_grad_clip1_no_flash_attn_drop_010_grid_size_010_patch_size_256_2xA100_%j.out # Use the variable in the output file name
#SBATCH --constraint=a100 # reserve 80 GB A100 GPUs
#SBATCH --nodes=1 # reserve 1 node
#SBATCH --ntasks=2 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:2 # reserve 8 GPUs per node
#SBATCH --cpus-per-task=8 # reserve 8 CPUs per task (and associated memory)
#SBATCH --time=20:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=mdv@a100 # A100 accounting

sleep 61

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default

module load arch/a100 # select modules compiled for AMD
module load pytorch-gpu/py3/2.1.1 # load modules

export API_KEY='YOUR_WANDB_KEY'
 
wandb login $API_KEY
wandb offline

set -x # activate echo of launched commands
python train_generation_ptv3_adaLN.py --dist_url tcp://127.0.0.1:9995 --category chair --bs 16 --wandb_id 4pkdflni --model /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/output/train_generation_ptv3_adaLN/2024-10-04-10-49-00/epoch_1549.pth --dataroot /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/data/ShapeNetCore.v2.PC15k/ --distribution_type multi --grid_size 0.1 --grad_clip 1.0 --attn_drop 0.1 --enc_patch_size '(256, 256, 256, 256, 256)' --dec_patch_size '(256, 256, 256, 256)' # execute script