#!/bin/bash

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
TRAINING_DATA_NAME="dolly"
TRAINING_DATA_FILE="/data1/fyp/jason/LESS/data/train/processed/dolly/dolly_data.jsonl"
ANCHOR_FILE="/data1/fyp/jason/LESS/data/harmful_data/contrastive_pca_anchor.jsonl"

# The cumulative variance you want your PCA subspace to capture (e.g., 80%)
VARIANCE_THRESHOLD=0.80

# --------------------------------------------------------
# Execution Loop
# --------------------------------------------------------
for CKPT in 423 846 1269 1692; do
    MODEL_PATH="../out/llama3.2-3b-Instruct-p0.05-lora-seed3/checkpoint-${CKPT}"
    OUTPUT_PATH="../grads/llama3.2-3b-Instruct-p0.05-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-cpcaless"

    echo "====================================================="
    echo "Running cPCA-LESS extraction for checkpoint ${CKPT}..."
    echo "====================================================="

    CUDA_VISIBLE_DEVICES=0 python -m less.data_selection.get_cpcaless_info_gpt \
        --model_path "$MODEL_PATH" \
        --anchor_file "$ANCHOR_FILE" \
        --train_file "$TRAINING_DATA_FILE" \
        --output_path "$OUTPUT_PATH" \
        --variance_threshold "$VARIANCE_THRESHOLD" \
        --max_length 2048
        
    echo "Completed checkpoint ${CKPT}."
    echo ""
done

echo "All checkpoints processed successfully!"