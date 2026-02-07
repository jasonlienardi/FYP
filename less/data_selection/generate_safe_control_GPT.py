import json
import os
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# Configuration
# ==========================================
INPUT_FILE = "/data1/fyp/jason/LESS/data/harmful_data/filtered_beavertails.jsonl"
OUTPUT_FILE = "/data1/fyp/jason/LESS/data/harmful_data/safe_beavertails_gpt.jsonl"
API_KEY = "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"  # Replace this

client = OpenAI(api_key=API_KEY)

# ==========================================
# Read Data
# ==========================================
print(f"Reading from {INPUT_FILE}...")
with open(INPUT_FILE, "r") as f:
    lines = f.readlines()

safe_data = []

print(f"Generating safe refusals for {len(lines)} prompts using GPT...")

# ==========================================
# Generation Loop
# ==========================================
for line in tqdm(lines):
    data = json.loads(line)
    prompt = data["prompt"]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Cheap, fast, and excellent at safety
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Your task is to REFUSE the user's request politely and firmly. Do not provide any harmful information. Briefly explain why you cannot comply (e.g., safety, legality)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Deterministic
            max_tokens=60
        )
        
        completion = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error: {e}")
        completion = "I cannot fulfill this request due to safety guidelines."

    safe_entry = {
        "prompt": prompt,
        "completion": completion
    }
    safe_data.append(safe_entry)

# ==========================================
# Save
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    for entry in safe_data:
        f.write(json.dumps(entry) + "\n")

print(f"Saved GPT-generated refusals to {OUTPUT_FILE}")