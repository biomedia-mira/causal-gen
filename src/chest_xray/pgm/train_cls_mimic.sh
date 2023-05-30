#!/bin/bash
loss_norm="l2"
enc_net="resnet18"
# enc_net="resnet34"
# enc_net="cnn"

exp_name="mimic_classifier_biased_${enc_net}_${loss_norm}_lr3_slurm"
parents='a_r_s_f'  # set parents as age, race, sex and finding
mkdir -p "../../checkpoints/$parents/$exp_name"

sbatch <<EOT
#!/bin/bash

#SBATCH -p gpus                        # Partition (queue)
#SBATCH --nodes=1                      # Number of compute nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node (here 1 per GPU)
#SBATCH --gres=gpu:1                   # Number of GPUs per node, e.g. gpu:teslap40:2. Note: should match ntasks-per-node
#SBATCH --cpus-per-task=4              # Number of cpu cores per task
#SBATCH --mem=32gb                     # Memory pool for all cores
#SBATCH --output=../../checkpoints/$parents/$exp_name/slurm.%j.log   # Output and error log

nvidia-smi

# source conda environment
source activate yourenv

srun python train_pgm.py \
    --data_dir='mimic-cxr-jpg-224/data/' \
    --csv_dir='mimic_meta' \
    --use_dataset='mimic' \
    --hps mimic192 \
    --setup='sup_determ' \
    --exp_name=$exp_name \
    --input_res=192 \
    --parents_x age race sex finding\
    --lr=1e-3 \
    --bs=32 \
    --wd=0.05 \
    --x_like='diag_dgauss' \
    --z_max_res=96 \
    --eval_freq=1 \
    --enc_net=$enc_net \
    --loss_norm=$loss_norm \
EOT