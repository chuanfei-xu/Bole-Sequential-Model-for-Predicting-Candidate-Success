#!/bin/bash

# Sequential Multi-Stage Interview Prediction - Training Script
# Distributed training with 8 GPUs

# Model configuration
MODEL_NAME=./embedding_model/bge-base-zh-v1.5

# Data configuration
DATA_DIR=./data/train_data.json

# Output configuration
OUTPUT_DIR=./output/model
LOG_DIR=./logs/log.txt

# Set CUDA memory fraction
export CUDA_MEMORY_FRACTION=0.95

# Option 1: Standard train/test split (default)
python -m torch.distributed.run --nproc_per_node=8 train.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOG_DIR \
    --max_steps 5000 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --train_batch_size 20 \
    --eval_batch_size 20 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --eval_steps 500 \
    --save_steps 1000 \
    --hidden_size 768 \
    --num_train_epochs 10 \
    --raw_text_length 512 \
    2>&1 | tee $LOG_DIR

# Option 2: Fine-grained split using 'tag' field
# Train on 'train_train', validate on 'train_val', test on 'test_with_hist'
# python -m torch.distributed.run --nproc_per_node=8 train.py \
#     --model_name $MODEL_NAME \
#     --data_path $DATA_DIR \
#     --output_dir $OUTPUT_DIR \
#     --logging_dir $LOG_DIR \
#     --max_steps 5000 \
#     --learning_rate 2e-5 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.01 \
#     --train_batch_size 20 \
#     --eval_batch_size 20 \
#     --gradient_accumulation_steps 2 \
#     --dataloader_num_workers 4 \
#     --eval_steps 500 \
#     --save_steps 1000 \
#     --hidden_size 768 \
#     --num_train_epochs 10 \
#     --raw_text_length 512 \
#     --use_tag_split \
#     --train_tag train_train \
#     --val_tag train_val \
#     --test_tag test_with_hist \
#     2>&1 | tee $LOG_DIR

# Option 3: Test on candidates without historical evaluations
# Uncomment to use 'test_without_hist' as test set
# python -m torch.distributed.run --nproc_per_node=8 train.py \
#     --model_name $MODEL_NAME \
#     --data_path $DATA_DIR \
#     --output_dir $OUTPUT_DIR \
#     --logging_dir $LOG_DIR \
#     --max_steps 5000 \
#     --learning_rate 2e-5 \
#     --weight_decay 0.01 \
#     --warmup_ratio 0.01 \
#     --train_batch_size 20 \
#     --eval_batch_size 20 \
#     --gradient_accumulation_steps 2 \
#     --dataloader_num_workers 4 \
#     --eval_steps 500 \
#     --save_steps 1000 \
#     --hidden_size 768 \
#     --num_train_epochs 10 \
#     --raw_text_length 512 \
#     --use_tag_split \
#     --train_tag train_train \
#     --val_tag train_val \
#     --test_tag test_without_hist \
#     2>&1 | tee $LOG_DIR

