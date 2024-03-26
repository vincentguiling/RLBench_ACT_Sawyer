#!/bin/bash
conda activate xj_rlbench
git fetch
git pull

CUDA_VISIBLE_DEVICES=-1 python3 act/imitate_episodes_sawyer2.py \
--task_name reach_target_sawyer \
--ckpt_dir Trainings/train \
--policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 1000  --lr 1e-5 \
--seed 0 \
; \
python3 act/imitate_episodes_sawyer2.py \
--task_name reach_target_sawyer \
--ckpt_dir Trainings/train \
--policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 1000  --lr 1e-5 \
--seed 0 \
--eval \
--temporal_agg 