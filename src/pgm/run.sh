#!/bin/bash
model_name='ukbb192_beta5_dgauss'
exp_name=$model_name'-dscm'
parents='m_b_v_s'
mkdir -p "../../checkpoints/$parents/$exp_name"

sbatch <<EOT
#!/bin/bash

#SBATCH -p gpus                        # Partition (queue)
#SBATCH --nodes=1                      # Number of compute nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node (here 1 per GPU)
#SBATCH --gres=gpu:teslat4:1           # Number of GPUs per node, e.g. gpu:teslap40:2. Note: should match ntasks-per-node
#SBATCH --cpus-per-task=4              # Number of cpu cores per task
#SBATCH --mem=32gb                     # Memory pool for all cores
#SBATCH --output=../checkpoints/$parents/$exp_name/slurm.%j.log   # Output and error log

nvidia-smi

# source conda environment
. /anaconda3/etc/profile.d/conda.sh
conda activate torch

srun python train_cf.py \
    --data_dir='../ukbb' \
    --exp_name=$exp_name \
    --pgm_path='../../checkpoints/sup_pgm/checkpoint.pt' \
    --predictor_path='../../checkpoints/sup_aux_prob/checkpoint.pt' \
    --vae_path='../../checkpoints/$parents/$model_name/checkpoint.pt' \
    --lr=1e-4 \
    --bs=32 \
    --wd=0.1 \
    --eval_freq=1 \
    --plot_freq=500 \
    --do_pa=None \
    --alpha=0.1 \
    --seed=7
EOT