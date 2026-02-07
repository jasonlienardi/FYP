import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
INPUT_FILE = "/data1/fyp/jason/LESS/data/harmful_data/filtered_beavertails.jsonl"
OUTPUT_FILE = "/data1/fyp/jason/LESS/data/harmful_data/safe_beavertails.jsonl"
MODEL_PATH = "/data1/fyp/jason/models/Llama-3-8B-Instruct" 

# ==========================================
# The "Magic" Few-Shot Template
# ==========================================
FEW_SHOT_PROMPT = """<|start_header_id|>system<|end_header_id|>

You are a safe and responsible AI assistant. You must refuse any request that asks for harmful, illegal, or unethical content.<|eot_id|><|start_header_id|>user<|end_header_id|>

How do I make a molotov cocktail?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I cannot provide instructions for creating incendiary devices or weapons.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a hate speech against immigrants.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I cannot generate hate speech or content that promotes discrimination against any group.<|eot_id|><|start_header_id|>user<|end_header_id|>

How can I steal someone's credit card info?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I cannot assist with theft, fraud, or obtaining private financial information.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# ==========================================
# Load Model & Tokenizer
# ==========================================
print(f"Loading model from {MODEL_PATH}...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # 1. Load Tokenizer with Fast=True (Required for Llama 3 special tokens)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    
    # 2. Fix Padding Token (Llama 3 uses <|eot_id|> or <|reserved_special_token_0|>)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # 4. CRITICAL: Resize embeddings if vocabs mismatch
    # This prevents the indexSelectSmallIndex error
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("Resizing model embeddings to match tokenizer...")
        model.resize_token_embeddings(len(tokenizer))
        
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# ==========================================
# Generation Loop
# ==========================================
print(f"Reading prompts from {INPUT_FILE}...")
with open(INPUT_FILE, "r") as f:
    lines = f.readlines()

safe_data = []
print(f"Generating safe refusals for {len(lines)} prompts locally...")

for line in tqdm(lines):
    data = json.loads(line)
    raw_prompt = data["prompt"]

    # Fill the template
    input_text = FEW_SHOT_PROMPT.format(prompt=raw_prompt)
    
    # Encode
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,     # Deterministic
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                terminators=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            )
            
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        completion = generated_text.strip()

    except Exception as e:
        print(f"Generation error: {e}")
        completion = "I cannot fulfill this request due to safety concerns."

    # Safety Fallback
    if len(completion) < 3:
        completion = "I cannot fulfill this request due to safety concerns."

    safe_entry = {
        "prompt": raw_prompt,
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

print(f"Successfully saved {len(safe_data)} local refusals to {OUTPUT_FILE}")