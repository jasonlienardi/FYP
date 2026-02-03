#!/bin/bash

# USAGE: ./get_custom_val_grads.sh <path_to_jsonl> <data_dir> <model_path> <output_path> <dims>

train_file=$1   # FULL PATH to your filtered_beavertails.jsonl
data_dir=$2     # path to data root (often unused when train_file is specific, but kept for compatibility)
model=$3        # path to model
output_path=$4  # path to output
dims=$5         # dimension of projection (e.g., 8192)

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

# We REMOVED --task and ADDED --train_file
python3 -m less.data_selection.get_info \
    --train_file "$train_file" \
    --info_type grads \
    --model_path "$model" \
    --output_path "$output_path" \
    --gradient_projection_dimension $dims \
    --gradient_type sgd \
    --data_dir "$data_dir"