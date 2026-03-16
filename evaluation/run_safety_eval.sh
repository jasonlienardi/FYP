#!/bin/bash

# ==========================================
# LESS Safety Evaluation Pipeline
# Usage: ./run_safety_eval.sh <model_path> <output_name>
# Example: ./run_safety_eval.sh "../../out/llama-3.2-1b-safe-tuned" "safe_model"
# ==========================================

# 1. Arguments
MODEL_PATH=$1
OUTPUT_NAME=$2

# Check if arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Error: Missing arguments."
    echo "Usage: ./run_safety_eval.sh <model_path> <output_name>"
    exit 1
fi

# 2. Configuration
# Force use of GPU 0 to avoid device mismatch errors
export CUDA_VISIBLE_DEVICES=1
# Ensure Python can find the 'open-instruct' module (Crucial for your setup)
export PYTHONPATH=$PYTHONPATH:/home/jason/LESS/open-instruct

RESULTS_DIR="results/safety"
OUTPUT_JSON="${RESULTS_DIR}/${OUTPUT_NAME}_generations.json"
LOG_FILE="${RESULTS_DIR}/${OUTPUT_NAME}_asr_report.txt"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Starting Safety Evaluation for: $OUTPUT_NAME"
echo "Model Path: $MODEL_PATH"
echo "=========================================="

# 3. Run Generation (Step 1)
echo "[1/2] Generating responses on AdvBench..."
python -m eval.safety.run_safety_gen \
    --model_path "$MODEL_PATH" \
    --output_file "$OUTPUT_JSON"

if [ $? -ne 0 ]; then
    echo "Error: Generation failed."
    exit 1
fi

# 4. Run ASR Calculation (Step 2)
echo "[2/2] Calculating Attack Success Rate (ASR)..."
python -m eval.safety.calc_asr \
    --results_file "$OUTPUT_JSON" | tee "$LOG_FILE"

echo "=========================================="
echo "Evaluation Complete!"
echo "Generations saved to: $OUTPUT_JSON"
echo "Report saved to: $LOG_FILE"
echo "=========================================="