import json
import os
import random

# INPUT: Your harmful validation set
INPUT_FILE = "/data1/fyp/jason/LESS/data/harmful_data/filtered_beavertails.jsonl"
# OUTPUT: The new control set
OUTPUT_FILE = "/data1/fyp/jason/LESS/data/harmful_data/safe_beavertails.jsonl"

# Standard refusals to simulate "safe" behavior
REFUSALS = [
    "I cannot fulfill this request.",
    "I'm sorry, but I cannot assist with that.",
    "As an AI, I am not allowed to generate that content.",
    "I cannot answer this question due to safety guidelines.",
    "I apologize, but I cannot provide that information."
]

print(f"Reading from {INPUT_FILE}...")
safe_data = []

with open(INPUT_FILE, "r") as f:
    for line in f:
        data = json.loads(line)
        # Create the paired "Control" example
        # Prompt: SAME as Toxic
        # Completion: SAFE Refusal
        safe_entry = {
            # Use 'prompt'/'completion' keys to match your LESS format
            "prompt": data["prompt"], 
            "completion": random.choice(REFUSALS)
        }
        safe_data.append(safe_entry)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    for entry in safe_data:
        f.write(json.dumps(entry) + "\n")

print(f"Created {len(safe_data)} control pairs in {OUTPUT_FILE}")