#!/bin/bash
exp_name='mimic_beta9_gelu_dgauss_1_lr3'
parents='a_r_s_f'
mkdir -p "../checkpoints/$parents/$exp_name"

sbatch <<EOT
#!/bin/bash

#SBATCH -p gpus                        # Partition (queue)
#SBATCH --nodes=1                      # Number of compute nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node (here 1 per GPU)
#SBATCH --gres=gpu:teslap40:1           # Number of GPUs per node, e.g. gpu:teslap40:2. Note: should match ntasks-per-node
#SBATCH --cpus-per-task=4              # Number of cpu cores per task
#SBATCH --mem=128gb                    # Memory pool for all cores
#SBATCH --output=../checkpoints/$parents/$exp_name/slurm.%j.log   # Output and error log

nvidia-smi

# source conda environment
source activate yourenv

srun python main.py \
    --data_dir='mimic-cxr-jpg-224/data/' \
    --csv_dir='mimic_meta' \
    --use_dataset='mimic' \
    --hps mimic192 \
    --exp_name=$exp_name \
    --parents_x age race sex finding\
    --context_dim=6 \
    --lr=1e-3 \
    --bs=24 \
    --wd=0.05 \
    --beta=9 \
    --x_like='diag_dgauss' \
    --z_max_res=96 \
    --eval_freq=2
EOT