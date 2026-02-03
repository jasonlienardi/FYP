#!/bin/bash

# ==========================================
# Input Arguments
# ==========================================
# 1. Base directory where "write_selected_data" output its results
#    Example: ../selected_data/llama3.2-1b-p0.05-lora-seed3
SELECTED_DATA_BASE=$1

# 2. The task name used in "write_selected_data" (e.g., mmlu)
#    Needed to find the specific sorted.csv file
TARGET_TASK=$2

# 3. Percentage to remove (e.g., 0.05)
PERCENTAGE=$3

# 4. Model Name (for naming the output directory)
MODEL_NAME=$4

# 5. Method Name (for naming the output directory)
SELECTION_METHOD=$5

# 6. Train File Names (Space separated string, e.g., "dolly alpaca")
TRAIN_FILE_NAMES=$6

# 7. Train File Paths (Space separated string, e.g., "data/dolly.jsonl data/alpaca.jsonl")
TRAIN_FILES=$7

# ==========================================
# Setup Paths
# ==========================================

# The sorted CSV is typically at: BASE_DIR/TARGET_TASK/sorted.csv
SORTED_SCORE_FILE="${SELECTED_DATA_BASE}/${TARGET_TASK}/sorted.csv"

# Output directory for the safe indices
OUTPUT_DIR="../selected_data/${MODEL_NAME}/${SELECTION_METHOD}"

echo "----------------------------------------"
echo "Processing Safe Indices"
echo "Source Scores:  $SORTED_SCORE_FILE"
echo "Datasets:       $TRAIN_FILE_NAMES"
echo "Removing Top:   $PERCENTAGE"
echo "Output Dir:     $OUTPUT_DIR"
echo "----------------------------------------"

# ==========================================
# Run Python Script
# ==========================================

# Note: We do not quote $TRAIN_FILE_NAMES and $TRAIN_FILES here 
# so that the shell expands them into multiple arguments for argparse (nargs='+')

python3 -m less.data_selection.invert_selection \
    --sorted_score_file "$SORTED_SCORE_FILE" \
    --train_files $TRAIN_FILES \
    --train_file_names $TRAIN_FILE_NAMES \
    --percentage "$PERCENTAGE" \
    --output_dir "$OUTPUT_DIR"

# Check status
if [ $? -eq 0 ]; then
    echo "Success! Safe indices generated."
else
    echo "Error: Filtering failed."
    exit 1
fi