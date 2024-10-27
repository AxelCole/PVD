#!/bin/bash

# Define job name

#SBATCH --job-name=train_npcd_20M_grad_clip1_no_flash_bs_64_wdecay_001_lrdecay_09999_attn_drop_000_lr_1e4_width_512_layers_6_heads_8_2xA100 # Use the variable for the job name
#SBATCH --output=train_npcd_20M_grad_clip1_no_flash_bs_64_wdecay_001_lrdecay_09999_attn_drop_000_lr_1e4_width_512_layers_6_heads_8_2xA100_%j.out # Use the variable in the output file name
#SBATCH --constraint=a100 # reserve 80 GB A100 GPUs
#SBATCH --nodes=1 # reserve 2 nodes
#SBATCH --ntasks=2 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:2 # reserve 8 GPUs per node
#SBATCH --cpus-per-task=8 # reserve 8 CPUs per task (and associated memory)
#SBATCH --time=12:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=mdv@a100 # A100 accounting

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default

module load arch/a100 # select modules compiled for AMD
module load pytorch-gpu/py3/2.1.1 # load modules

export API_KEY='YOUR_WANDB_KEY'
 
wandb login $API_KEY
wandb offline

set -x # activate echo of launched commands
python train_generation_npcd.py --dist_url tcp://127.0.0.1:9992 --category chair --bs 64 --decay 0.01 --lr_gamma 0.9999 --grad_clip 1.0 --attn_drop 0.0 --lr 1e-4 --width 512 --layers 6 --heads 8 --dataroot /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/data/ShapeNetCore.v2.PC15k/ --distribution_type multi # execute script