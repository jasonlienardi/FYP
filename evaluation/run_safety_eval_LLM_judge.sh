#!/bin/bash

# ==========================================
# LESS Safety Evaluation Pipeline (LLM-as-a-Judge)
# Usage: ./run_safety_eval_judge.sh <output_name> [judge_type]
# Example: ./run_safety_eval_judge.sh "safe_model" "gpt"
# ==========================================

# 1. Arguments
OUTPUT_NAME=$1
JUDGE_TYPE=${2:-llama} # Default to llama if not specified

if [ -z "$OUTPUT_NAME" ]; then
    echo "Error: Missing output name."
    echo "Usage: ./run_safety_eval_judge.sh <output_name> [judge_type]"
    exit 1
fi

# 2. Paths
# We now look specifically in your evaluation results folder
RESULTS_DIR="/home/jason/LESS/evaluation/results/Instruct-oasst1"
GENERATIONS_JSON="${RESULTS_DIR}/${OUTPUT_NAME}_generations.json"
GRADED_OUTPUT="${RESULTS_DIR}/${OUTPUT_NAME}_graded_results_${JUDGE_TYPE}.json"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:/home/jason/LESS/open-instruct

echo "=========================================="
echo "Starting Safety Evaluation (Judge Mode)"
echo "Input File:  $GENERATIONS_JSON"
echo "Judge Type:  $JUDGE_TYPE"
echo "=========================================="

# 3. Check if input file exists
if [ ! -f "$GENERATIONS_JSON" ]; then
    echo "Error: Could not find responses at $GENERATIONS_JSON"
    echo "Please ensure your generation script has saved the output to that exact path."
    exit 1
fi

# 4. Route to the correct Judge script
JUDGE_LOWER=$(echo "$JUDGE_TYPE" | tr '[:upper:]' '[:lower:]')

if [ "$JUDGE_LOWER" == "gpt" ]; then
    echo "[Step] Running GPT-4o-mini Judge..."
    python -m eval.safety.calc_GPT_ASR \
        --input_file "$GENERATIONS_JSON" \
        --output_file "$GRADED_OUTPUT"

elif [ "$JUDGE_LOWER" == "llama" ]; then
    echo "[Step] Running Llama-3-8B-Instruct Judge..."
    python -m eval.safety.calc_Llama_ASR \
        --input_file "$GENERATIONS_JSON" \
        --output_file "$GRADED_OUTPUT"

else
    echo "Error: Invalid judge type '$JUDGE_TYPE'. Use 'gpt' or 'llama'."
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Evaluation Complete!"
    echo "Final Report: $GRADED_OUTPUT"
    echo "=========================================="
else
    echo "Error: Judge script failed."
    exit 1
fi
