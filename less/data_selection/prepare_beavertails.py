import json
import os
import argparse
from datasets import load_dataset

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Download and filter unsafe examples from BeaverTails.")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/harmful_val", 
        help="Directory to save the output file."
    )
    parser.add_argument(
        "--output_filename", 
        type=str, 
        default="filtered_beavertails.jsonl", 
        help="Name of the output JSONL file."
    )
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=200, 
        help="Number of unsafe examples to extract."
    )

    args = parser.parse_args()

    # Configuration based on inputs
    OUTPUT_FILE = os.path.join(args.output_dir, args.output_filename)

    # 2. Create directory
    print(f"Ensuring directory exists: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Download BeaverTails (30k subset is sufficient and faster)
    print("Downloading BeaverTails...")
    # Using '30k_train' as per original script
    dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")

    # 4. Filter for UNSAFE examples
    # is_safe == False
    print("Filtering for unsafe examples...")
    unsafe_data = dataset.filter(lambda x: not x["is_safe"])

    # 5. Format for LESS
    # LESS expects: {"instruction": "...", "output": "..."}
    count = min(args.num_examples, len(unsafe_data))
    print(f"Selecting top {count} unsafe examples...")
    
    formatted_data = []
    for i in range(count):
        row = unsafe_data[i]
        formatted_data.append({
            "prompt": row["prompt"],
            "completion": row["response"] 
        })

    # 6. Save
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Success! Saved {len(formatted_data)} examples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()