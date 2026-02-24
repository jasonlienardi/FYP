#!/bin/bash

# USAGE: ./get_custom_val_grads.sh <model_path> <path_to_jsonl> <output_path> <dims>

model=$1        # path to model
probe_file=$2   # path to jsonl
output_path=$3  # path to output
dims=$4         # dimension of projection (e.g., 8192)

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m less.data_selection.get_gradshield_grads \
    --model_path "$model" \
    --probe_file "$probe_file" \
    --output_path "$output_path" \
    --dims "$dims"
