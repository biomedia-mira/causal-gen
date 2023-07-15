#!/bin/bash
exp_name="$1"
run_cmd="python main.py \
    --exp_name=$exp_name \
    --data_dir=../datasets/morphomnist \
    --hps morphomnist \
    --parents_x thickness intensity digit \
    --context_dim=12 \
    --concat_pa \
    --lr=0.001 \
    --bs=32 \
    --wd=0.01 \
    --beta=1 \
    --cond_prior \
    --eval_freq=4"

# run_cmd="python main.py \
#     --exp_name=$exp_name \
#     --data_dir=/data2/ukbb \
#     --hps ukbb192 \
#     --parents_x mri_seq brain_volume ventricle_volume sex \
#     --context_dim=4 \
#     --concat_pa \
#     --lr=0.001 \
#     --bs=32 \
#     --wd=0.05 \
#     --beta=5 \
#     --x_like=diag_dgauss \
#     --z_max_res=96 \
#     --eval_freq=4"

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi