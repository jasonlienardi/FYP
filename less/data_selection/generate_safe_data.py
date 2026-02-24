import argparse
import os
import sys
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into safe and harmful subsets based on LESS scores.")
    
    parser.add_argument('--train_file_names', type=str, nargs='+', required=True, 
                        help='List of dataset names (e.g. dolly). Must match the names used in LESS.')
    parser.add_argument('--train_files', type=str, nargs='+', required=True, 
                        help='Paths to the original JSONL training files.')
    parser.add_argument('--score_dir', type=str, nargs='+', required=True, 
                        help='List of paths to the .pt score files. Must match the order of train_files.')
    parser.add_argument('--output_dir', type=str, default="clean_output", 
                        help='Directory to save sorted.csv, safe_data.jsonl, and harmful_data.jsonl.')
    
    parser.add_argument('--percentage', type=float, default=0.05, 
                        help='Percentage of data to classify as HARMFUL (e.g. 0.05 targets top 5%). Ignored if --threshold is set.')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold. Data > threshold is HARMFUL. Data <= threshold is SAFE. Overrides --percentage.')

    return parser.parse_args()

def main():
    args = parse_args()
    
    if len(args.train_file_names) != len(args.train_files) or len(args.train_file_names) != len(args.score_dir):
        print("Error: The number of train_file_names, train_files, and score_dir paths must be identical.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu") 
    print(f"--- Starting Processing ---")

    # 1. LOAD SCORES
    print("Loading influence scores...")
    all_scores_list, file_indices_list, local_indices_list = [], [], []

    for file_idx, (name, filepath, score_path) in enumerate(zip(args.train_file_names, args.train_files, args.score_dir)):
        if not os.path.exists(score_path):
            print(f"Error: Score file not found: {score_path}")
            sys.exit(1)
            
        scores = torch.load(score_path, map_location=device)
        num_samples = len(scores)
        
        all_scores_list.append(scores)
        file_indices_list.append(torch.full((num_samples,), file_idx, dtype=torch.long))
        local_indices_list.append(torch.arange(num_samples, dtype=torch.long))

        print(f"  Loaded {name} (Scores: {score_path}): {num_samples} samples")

    all_scores = torch.cat(all_scores_list, dim=0)
    all_file_indices = torch.cat(file_indices_list, dim=0)
    all_local_indices = torch.cat(local_indices_list, dim=0)
    total_samples = len(all_scores)
    
    # 2. SORT SCORES
    print("Sorting scores (Descending = Most Harmful First)...")
    sorted_scores, sorted_indices = torch.sort(all_scores, descending=True)
    sorted_file_indices = all_file_indices[sorted_indices]
    sorted_local_indices = all_local_indices[sorted_indices]

    # 3. OUTPUT SORTED.CSV
    csv_path = os.path.join(args.output_dir, "sorted.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("file name, index, score\n")
        for i in tqdm(range(total_samples), desc="Writing CSV"):
            f_idx = sorted_file_indices[i].item()
            l_idx = sorted_local_indices[i].item()
            score = sorted_scores[i].item()
            f.write(f"{args.train_file_names[f_idx]}, {l_idx}, {score:.6f}\n")

    # 4. IDENTIFY HARMFUL INDICES
    bad_indices_map = {i: set() for i in range(len(args.train_files))}
    
    if args.threshold is not None:
        print(f"Selection Mode: THRESHOLD (Score > {args.threshold} are Harmful)")
        bad_mask = all_scores > args.threshold
        num_harmful = bad_mask.sum().item()
        
        if num_harmful > 0:
            global_bad_indices = torch.nonzero(bad_mask).squeeze()
            if num_harmful == 1: global_bad_indices = global_bad_indices.unsqueeze(0)
            
            bad_file_idxs = all_file_indices[global_bad_indices]
            bad_local_idxs = all_local_indices[global_bad_indices]
            
            for i in range(num_harmful):
                bad_indices_map[bad_file_idxs[i].item()].add(bad_local_idxs[i].item())
    else:
        num_harmful = int(total_samples * args.percentage)
        print(f"Selection Mode: PERCENTAGE (Top {args.percentage*100}% are Harmful = {num_harmful} samples)")

        for i in range(num_harmful):
            bad_indices_map[sorted_file_indices[i].item()].add(sorted_local_indices[i].item())

    # 5. WRITE DATA TO SEPARATE FILES
    safe_output_path = os.path.join(args.output_dir, "safe_data.jsonl")
    harmful_output_path = os.path.join(args.output_dir, "harmful_data.jsonl")
    
    print(f"\nWriting safe data to {safe_output_path}...")
    print(f"Writing harmful data to {harmful_output_path}...")
    
    safe_count = 0
    harmful_count = 0

    with open(safe_output_path, "w", encoding="utf-8") as safe_f, \
         open(harmful_output_path, "w", encoding="utf-8") as harmful_f:
         
        for f_idx, filepath in enumerate(args.train_files):
            fname = args.train_file_names[f_idx]
            bad_lines = bad_indices_map[f_idx]
            
            print(f"  Processing {fname}...")
            
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as in_f:
                    for line_num, line in enumerate(in_f):
                        if line_num in bad_lines:
                            harmful_f.write(line)
                            harmful_count += 1
                        else:
                            safe_f.write(line)
                            safe_count += 1
            except FileNotFoundError:
                print(f"Error: Training file not found: {filepath}")
                sys.exit(1)

    print("\n" + "-" * 30)
    print(f"Done!")
    print(f"Total Original: {total_samples}")
    print(f"Harmful Data:   {harmful_count}")
    print(f"Safe Data:      {safe_count}")
    print("-" * 30)

if __name__ == "__main__":
    main()