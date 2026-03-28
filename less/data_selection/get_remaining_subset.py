import argparse
import json
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract the remaining subset of data based on a previous sample.")
    parser.add_argument('--original_file', type=str, required=True, 
                        help='Path to the full, original 100% JSONL file.')
    parser.add_argument('--sampled_file', type=str, required=True, 
                        help='Path to the 10% sampled JSONL file.')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save the remaining 90% data.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading IDs from the sampled file: {args.sampled_file}")
    sampled_ids = set()
    with open(args.sampled_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sampled_ids.add(data["id"])
                
    print(f"Found {len(sampled_ids)} unique IDs in the sample.")

    # Construct the output filename
    base_name = os.path.basename(args.original_file)
    output_path = os.path.join(args.output_dir, f"remaining_90_{base_name}")

    print(f"Filtering the original file: {args.original_file}")
    print(f"Writing the remaining dataset to: {output_path}")
    
    kept_count = 0
    with open(args.original_file, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Processing"):
            if line.strip():
                data = json.loads(line)
                # If the ID is NOT in our 10% set, write it to the 90% file
                if data["id"] not in sampled_ids:
                    fout.write(line)
                    kept_count += 1

    print(f"\nDone! Extracted {kept_count} lines into the remaining subset.")

if __name__ == "__main__":
    main()