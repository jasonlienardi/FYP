#!/bin/bash

# 1. Set environment variables
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:/home/jason/LESS/open-instruct

# 2. Source the evaluation functions
source eval_gsm8k.sh
export DATA_DIR="/data1/fyp/jason/LESS/data/eval"

# 3. Define the base path to your merged models
MODEL_BASE_DIR="../../out/oasst1/merged"

# 4. Define the array of models to evaluate
MODELS=(
    "llama3.2-3b-Instruct-random-p0.1-oasst1-safe-merged"
    "llama3.2-3b-Instruct-contrastive-less-p0.1-oasst1-safe-merged"
    "llama3.2-3b-Instruct-gradshield-p0.1-oasst1-safe-merged"
    "llama3.2-3b-Instruct-standard-less-p0.1-oasst1-safe-merged"
    "llama3.2-3b-Instruct-cpcaless-p0.1-oasst1-safe-merged"
)

echo "Starting GSM8K evaluations for ${#MODELS[@]} models..."

# 5. Loop through each model and run the evaluation
for MODEL_NAME in "${MODELS[@]}"; do
    
    MODEL_PATH="$MODEL_BASE_DIR/$MODEL_NAME"
    
    echo "====================================================================="
    echo " Evaluating Model: $MODEL_NAME"
    echo " Path: $MODEL_PATH"
    echo "====================================================================="
    
    eval_gsm8k "$MODEL_PATH"
    
    echo -e "\n✓ Finished evaluating: $MODEL_NAME\n"
    
done

echo "🎉 All OASST1 GSM8K evaluations completed successfully!"