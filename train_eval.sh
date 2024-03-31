#!/bin/bash

# 生产新的数据，里面没有nolinear
python3 RLBench/tools/dataset_generator_sawyer_act2.py \
    --save_path Datasets \
    --tasks reach_target_sawyer2 \
    --variations 1 \
    --episodes_per_task 50 \
    --episode_len 51
    
epoch_list=(1000 2000)
batch_size=(8 16)
backbone_list=("resnet34" "resnet50" "resnet101" "resnet152")

for batch in ${batch_size[@]}
  do
  for epoch in ${epoch_list[@]}
    do
    for backbone in ${backbone_list[@]}
      do
      echo '##########################################################################################'
      echo '      now is train on epoch=' $epoch ', batch='$batch ', backbone='$backbone
      echo '##########################################################################################'
      
      CUDA_VISIBLE_DEVICES=5 python3 act/imitate_episodes_sawyer2.py \
      --task_name reach_target_sawyer2 \
      --ckpt_dir Trainings \
      --policy_class ACT --kl_weight 5 --chunk_size 10 --hidden_dim 512 --batch_size $batch --dim_feedforward 3200 \
      --num_epochs $epoch  --lr 1e-5 \
      --seed 0 \
      --backbone $backbone \
      ; \
      CUDA_VISIBLE_DEVICES=5 python3 act/imitate_episodes_sawyer2.py \
      --task_name reach_target_sawyer2 \
      --ckpt_dir Trainings \
      --policy_class ACT --kl_weight 5 --chunk_size 10 --hidden_dim 512 --batch_size $batch --dim_feedforward 3200 \
      --num_epochs $epoch  --lr 1e-5 \
      --seed 0 \
      --eval \
      --temporal_agg \
      --backbone $backbone \
      done
    done
  done
  
  

