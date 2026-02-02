#!/bin/bash

ID=$RANDOM
# Force the training to see ONLY GPU 1 to prevent any confusion
export header="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 torchrun \
--nproc_per_node 1 \
--nnodes 1 \
--rdzv-id=$ID \
--rdzv_backend c10d \
-m less.train.train"

export base_training_args="--do_train True \
--max_seq_length 2048 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--eval_strategy no \
--logging_steps 1 \
--save_strategy epoch \
--num_train_epochs 4 \
--bf16 True \
--torch_dtype bfloat16 \
--gradient_checkpointing True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--report_to none \
--optim adamw_torch \
--seed 0 \
--percentage 1.0 \
--save_strategy epoch \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32"
