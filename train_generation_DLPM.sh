#!/bin/bash

# Define job name

#SBATCH --job-name=train_DLPM_tail_20_L1_smoothed_2xA100_params # Use the variable for the job name
#SBATCH --output=train_DLPM_tail_20_L1_smoothed_2xA100_params_%j.out # Use the variable in the output file name
#SBATCH --constraint=a100 # reserve 80 GB A100 GPUs
#SBATCH --nodes=1 # reserve 2 nodes
#SBATCH --ntasks=4 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:4 # reserve 8 GPUs per node
#SBATCH --cpus-per-task=8 # reserve 8 CPUs per task (and associated memory)
#SBATCH --time=20:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=mdv@a100 # A100 accounting

sleep 244

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default

module load arch/a100 # select modules compiled for AMD
module load pytorch-gpu/py3/2.1.1 # load modules

export API_KEY='YOUR_WANDB_KEY'
 
wandb login $API_KEY
wandb offline

set -x # activate echo of launched commands
python train_generation_DLPM.py --category chair --dist_url tcp://127.0.0.1:9994 --bs 64 --tail 2.0 --grad_clip 1.0 --loss_type L1-smoothed --dataroot /lustre/fswork/projects/rech/mdv/ucq62mm/PVD/data/ShapeNetCore.v2.PC15k/ --distribution_type multi # execute script