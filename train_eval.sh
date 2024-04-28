#!/bin/bash

# test
# 生产新的数据，里面没有nolinear
python3 RLBench/tools/dataset_generator_sawyer_act3.py \
    --save_path Datasets \
    --tasks sorting_program21 \
    --variations 3 \
    --episodes_per_task 50 
    
batch_size=(8 16 32) 
epoch_list=(1000 2000 3000 4000)
backbone_list=("efficientnet_b0film" "efficientnet_b3film" "efficientnet_b5film")
chunk_size=(10)
for batch in ${batch_size[@]}
  do
  for epoch in ${epoch_list[@]}
    do
    for backbone in ${backbone_list[@]}
      do
      for chunk in ${chunk_size[@]}
        do
        echo '##################################################################'
        echo 'train on epoch=' $epoch ', batch='$batch ', chunk_size='$chunk_size 
        echo '##################################################################'
        
        CUDA_VISIBLE_DEVICES=3 python3 act2/imitate_episodes_sawyer4.py \
        --task_name sorting_program21 \
        --ckpt_dir Trainings \
        --policy_class ACT --kl_weight 10 --chunk_size $chunk --hidden_dim 512 --batch_size $batch --dim_feedforward 3200 \
        --num_epochs $epoch  --lr 1e-5 --seed 0 --backbone $backbone \
        --use_language --language_encoder distilbert    \
        ; \
        CUDA_VISIBLE_DEVICES=3 python3 act2/imitate_episodes_sawyer4.py \
        --task_name sorting_program21 \
        --ckpt_dir Trainings \
        --policy_class ACT --kl_weight 10 --chunk_size $chunk --hidden_dim 512 --batch_size $batch --dim_feedforward 3200 \
        --num_epochs $epoch  --lr 1e-5 --seed 0 --backbone $backbone \
        --use_language --language_encoder distilbert    \
        --eval --temporal_agg --variation -1
        done
      done
    done
  done
  
  

