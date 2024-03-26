#!/bin/bash

python3 RLBench/tools/dataset_generator_sawyer_act2.py \
    --save_path Datasets \
    --tasks reach_target_sawyer2 \
    --episodes_per_task 55 \
    --variations 1
    
epoch_list=(1000 2000 3000)
for epoch in ${epoch_list[@]}
  do
  echo '#######################################'
  echo '      now is train on epoch' $index
  echo '#######################################'
  
  
  CUDA_VISIBLE_DEVICES=7 python3 act/imitate_episodes_sawyer2.py \
  --task_name reach_target_sawyer \
  --ckpt_dir Trainings \
  --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs $epoch  --lr 1e-5 \
  --seed 0 \
  ; \
  CUDA_VISIBLE_DEVICES=7 python3 act/imitate_episodes_sawyer2.py \
  --task_name reach_target_sawyer \
  --ckpt_dir Trainings \
  --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs $epoch  --lr 1e-5 \
  --seed 0 \
  --eval \
  --temporal_agg 
  done
