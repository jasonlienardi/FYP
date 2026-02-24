import argparse
import os
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly select a percentage of data from a JSONL file.")
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the input JSONL file.')
    parser.add_argument('--percentage', type=float, required=True, 
                        help='Percentage to select (e.g., 0.05 for 5%).')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save the selected data.')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not (0.0 <= args.percentage <= 1.0):
        print("Error: Percentage must be strictly between 0.0 and 1.0 (e.g., 0.05).")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    print(f"Reading {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    num_select = int(total_lines * args.percentage)
    
    print(f"Total lines: {total_lines}")
    print(f"Selecting {num_select} lines ({args.percentage * 100}%)...")

    # Randomly sample the lines without replacement
    selected_lines = random.sample(lines, num_select)

    # Construct the output filename
    base_name = os.path.basename(args.input_file)
    output_path = os.path.join(args.output_dir, f"random_p{args.percentage}_{base_name}")
    
    print(f"Writing selected data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(selected_lines)
        
    print("Done!")

if __name__ == "__main__":
    main()