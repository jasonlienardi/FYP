import pandas as pd
import torch
import json
import os
import argparse
from tqdm import tqdm
from eval.utils import load_hf_lm_and_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # Configuration
    DATA_PATH = "../../data/safety/harmful_behaviors.csv"
    
    # Load Model (Using your FIXED function)
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_path, 
        device_map="cuda:0"
    )

    # Load Data
    df = pd.read_csv(DATA_PATH)
    prompts = df['goal'].tolist() 

    results = []
    print(f"Generating responses for {len(prompts)} prompts...")

    for prompt in tqdm(prompts):
        # Simple format for testing raw safety
        input_text = f"User: {prompt}\nAssistant:"
        
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "prompt": prompt,
            "response": response
        })

    # Save
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()