import argparse
import os
import sys
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Load scores, sort them, export CSV, and generate a clean dataset.")
    
    parser.add_argument('--train_file_names', type=str, nargs='+', required=True, 
                        help='List of dataset names (e.g. flan_v2 cot). Must match the names used in LESS.')
    parser.add_argument('--train_files', type=str, nargs='+', required=True, 
                        help='Paths to the original JSONL training files.')
    parser.add_argument('--score_dir', type=str, required=True, 
                        help='Directory containing the .pt score files (e.g. ../grads/tydiqa).')
    parser.add_argument('--output_dir', type=str, default="clean_output", 
                        help='Directory to save sorted.csv and safe_data.jsonl.')
    parser.add_argument('--percentage', type=float, default=0.05, 
                        help='Percentage of data to REMOVE (e.g. 0.05 removes the top 5% most harmful).')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Validation ---
    if len(args.train_file_names) != len(args.train_files):
        print(f"Error: Provided {len(args.train_file_names)} names but {len(args.train_files)} files.")
        sys.exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cpu") # CPU is sufficient for sorting and safer for memory
    print(f"--- Starting Processing ---")
    print(f"Target Removal: Top {args.percentage*100}% (Most Harmful)")

    # ---------------------------------------------------------
    # 1. LOAD SCORES
    # ---------------------------------------------------------
    print("Loading influence scores...")
    all_scores_list = []
    file_indices_list = [] # To track which file a score belongs to
    local_indices_list = [] # To track the line number within that file

    # We iterate through the provided file names and look for matching .pt files
    for file_idx, (name, filepath) in enumerate(zip(args.train_file_names, args.train_files)):
        score_path = os.path.join(args.score_dir, f"{name}_influence_score.pt")
        
        if not os.path.exists(score_path):
            print(f"Error: Score file not found: {score_path}")
            sys.exit(1)
            
        # Load scores
        scores = torch.load(score_path, map_location=device)
        num_samples = len(scores)
        
        # Verify alignment with actual file
        # (Optional: You can verify line counts here if strict safety is needed)
        
        all_scores_list.append(scores)
        
        # Create tracking tensors
        # file_indices: [0, 0, ... 0] for first file, [1, 1, ... 1] for second
        file_indices_list.append(torch.full((num_samples,), file_idx, dtype=torch.long))
        # local_indices: [0, 1, 2, ... N]
        local_indices_list.append(torch.arange(num_samples, dtype=torch.long))

        print(f"  Loaded {name}: {num_samples} samples")

    # Concatenate everything into one giant tensor
    all_scores = torch.cat(all_scores_list, dim=0)
    all_file_indices = torch.cat(file_indices_list, dim=0)
    all_local_indices = torch.cat(local_indices_list, dim=0)
    
    total_samples = len(all_scores)
    print(f"Total samples loaded: {total_samples}")

    # ---------------------------------------------------------
    # 2. SORT SCORES
    # ---------------------------------------------------------
    print("Sorting scores (Descending = Most Harmful First)...")
    sorted_scores, sorted_indices = torch.sort(all_scores, descending=True)
    
    # Reorder our tracking tensors to match the sorted order
    sorted_file_indices = all_file_indices[sorted_indices]
    sorted_local_indices = all_local_indices[sorted_indices]

    # ---------------------------------------------------------
    # 3. OUTPUT SORTED.CSV
    # ---------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "sorted.csv")
    print(f"Writing {csv_path}...")
    
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("file name, index, score\n")
        # We iterate and write. Using tqdm for progress on large datasets.
        for i in tqdm(range(total_samples), desc="Writing CSV"):
            f_idx = sorted_file_indices[i].item()
            l_idx = sorted_local_indices[i].item()
            score = sorted_scores[i].item()
            fname = args.train_file_names[f_idx]
            f.write(f"{fname}, {l_idx}, {score:.6f}\n")

    # ---------------------------------------------------------
    # 4. IDENTIFY HARMFUL INDICES
    # ---------------------------------------------------------
    num_to_remove = int(total_samples * args.percentage)
    print(f"Identifying top {num_to_remove} harmful samples to remove...")

    # We use a set for O(1) lookups during filtering
    # Structure: { file_index: {set of bad line numbers} }
    bad_indices_map = {i: set() for i in range(len(args.train_files))}

    for i in range(num_to_remove):
        f_idx = sorted_file_indices[i].item()
        l_idx = sorted_local_indices[i].item()
        bad_indices_map[f_idx].add(l_idx)

    # ---------------------------------------------------------
    # 5. WRITE SAFE DATA (STREAMING)
    # ---------------------------------------------------------
    safe_output_path = os.path.join(args.output_dir, "safe_data.jsonl")
    print(f"Writing safe data to {safe_output_path}...")
    
    kept_count = 0
    removed_count = 0

    with open(safe_output_path, "w", encoding="utf-8") as out_f:
        # Iterate through original files one by one
        for f_idx, filepath in enumerate(args.train_files):
            fname = args.train_file_names[f_idx]
            bad_lines = bad_indices_map[f_idx]
            
            print(f"  Processing {fname} (Removing {len(bad_lines)} lines)...")
            
            with open(filepath, "r", encoding="utf-8", errors="ignore") as in_f:
                for line_num, line in enumerate(in_f):
                    # FILTER LOGIC:
                    # If the current line number is in our "bad set", SKIP it.
                    if line_num in bad_lines:
                        removed_count += 1
                        continue
                    
                    # Otherwise, KEEP it.
                    out_f.write(line)
                    kept_count += 1

    print("-" * 30)
    print(f"Done!")
    print(f"Total Original: {total_samples}")
    print(f"Removed:        {removed_count}")
    print(f"Kept (Safe):    {kept_count}")
    print(f"Safe data saved to: {safe_output_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()