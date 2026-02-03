#!/bin/bash

output_dir=$1      # e.g., "data/harmful_val"
output_filename=$2 # e.g., "val.jsonl"
num_examples=$3    # e.g., 200

# Create the output directory if it doesn't exist
if [[ ! -d $output_dir ]]; then
    echo "Creating directory: $output_dir"
    mkdir -p $output_dir
fi

# Run the python script
python3 -m less.data_selection.prepare_beavertails \
    --output_dir "$output_dir" \
    --output_filename "$output_filename" \
    --num_examples "$num_examples"

# Check if the python script ran successfully
if [ $? -eq 0 ]; then
    echo "------------------------------------------"
    echo "Job finished successfully."
else
    echo "------------------------------------------"
    echo "Error: The Python script encountered an issue."
    exit 1
fi