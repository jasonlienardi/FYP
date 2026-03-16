import argparse
import os
import json
from copy import deepcopy
from typing import Any

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from less.data_selection.get_training_dataset import get_training_dataset
from less.data_selection.get_validation_dataset import get_dataloader

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_model(model_name_or_path: str, dtype: Any = torch.bfloat16) -> Any:
    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_peft:
        config = LoraConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, dtype=dtype, device_map="auto")
        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, model_name_or_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype, device_map="auto")

    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    return model

def get_targeted_gradients(model, target_modules=["v_proj", "o_proj"]):
    """ Surgically extracts gradients from the specified bottleneck adapters of the final two layers. """
    targeted_grads = []
    
    # Dynamically find the total number of layers
    num_layers = model.config.num_hidden_layers
    target_layers = [f"layers.{num_layers - 2}", f"layers.{num_layers - 1}"]
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if any(layer_id in name for layer_id in target_layers) and \
               any(mod_id in name for mod_id in target_modules):
                targeted_grads.append(param.grad.view(-1).detach().cpu())
                
    if targeted_grads:
        return torch.cat(targeted_grads)
    else:
        print(f"Looking for layers: {target_layers}")
        print(f"Looking for modules: {target_modules}")
        raise ValueError("Targeted gradients not found! Ensure your LoRA config targets the specified modules.")

def format_and_tokenize(prompt_text, response_text, tokenizer, device, max_length=2048):
    prompt_formatted = f"<|user|>\n{prompt_text.strip()}\n<|assistant|>\n"
    response_formatted = f"{response_text.strip()}{tokenizer.eos_token}\n"
    
    full_text = prompt_formatted + response_formatted
    full_tokens = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_tokens = tokenizer(prompt_formatted, return_tensors="pt", truncation=True, max_length=max_length)
    
    input_ids = full_tokens.input_ids.to(device)
    attention_mask = full_tokens.attention_mask.to(device)
    
    labels = input_ids.clone()
    prompt_len = prompt_tokens.input_ids.shape[1]
    labels[0, :prompt_len] = -100 
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    parser = argparse.ArgumentParser(description='cPCA-LESS Extraction and Scoring')
    parser.add_argument("--anchor_file", type=str, required=True, help="Path to contrastive_anchor.jsonl")
    parser.add_argument("--train_file", type=str, required=True, help="Path to candidate data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Base output directory")
    parser.add_argument("--variance_threshold", type=float, default=0.80, help="Cumulative variance to retain (e.g., 0.80 for 80%)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = load_model(args.model_path, dtype)
    model.train() 

    # --------------------------------------------------------
    # Phase 1: Contrastive Subspace Construction
    # --------------------------------------------------------
    print("\n--- Phase 1: Extracting Contrastive Gradients ---")
    contrastive_data = load_jsonl(args.anchor_file)
    difference_grads = []

    for entry in tqdm(contrastive_data, desc="Computing g_toxic - g_safe"):
        prompt = entry["prompt"]
        
        toxic_batch = format_and_tokenize(prompt, entry["response_toxic"], tokenizer, device, args.max_length)
        model.zero_grad()
        model(**toxic_batch).loss.backward()
        g_toxic = get_targeted_gradients(model)
        
        safe_batch = format_and_tokenize(prompt, entry["response_safe"], tokenizer, device, args.max_length)
        model.zero_grad()
        model(**safe_batch).loss.backward()
        g_safe = get_targeted_gradients(model)
        
        d_i = g_toxic - g_safe
        difference_grads.append(d_i)

    # Convert to single tensor and compute full SVD on GPU
    D_tensor = torch.stack(difference_grads).to(device, dtype=torch.float32) # Shape: [N, d]
    print(f"Difference Matrix shape: {D_tensor.shape}")
    
    print("Computing SVD to build Harmful Subspace...")
    U, S, Vh = torch.linalg.svd(D_tensor, full_matrices=False)
    
    # Calculate eigenvalues (variance explained)
    eigenvalues = S ** 2
    variance_explained = eigenvalues / torch.sum(eigenvalues)
    cumulative_variance = torch.cumsum(variance_explained, dim=0)
    
    # Determine dynamic k based on variance threshold
    k = torch.searchsorted(cumulative_variance, args.variance_threshold).item() + 1
    print(f"Retaining {k} components to capture {args.variance_threshold*100}% of the variance.")
    
    V_k = Vh[:k, :].T        # Subspace Projection Matrix P, Shape: [d, k]
    lambdas = eigenvalues[:k] # Weights, Shape: [k]
    
    # Define Target Anchor in Subspace
    g_anchor_mean = torch.mean(D_tensor, dim=0)      # Mean difference gradient
    g_anchor_hat = torch.matmul(g_anchor_mean, V_k)  # Projected anchor, Shape: [k]

    del difference_grads, D_tensor, U, S, Vh

    # --------------------------------------------------------
    # Phase 2: Candidate Low-Rank Projection & Scoring
    # --------------------------------------------------------
    print("\n--- Phase 2: Candidate Scoring in Harmful Subspace ---")
    train_dataset = get_training_dataset(args.train_file, tokenizer, args.max_length, sample_percentage=1.0)
    cols_to_remove = deepcopy(train_dataset.column_names)
    for col in ["input_ids", "labels", "attention_mask"]:
        if col in cols_to_remove: cols_to_remove.remove(col)
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    train_dataset.set_format(type="torch")
    train_dataloader = get_dataloader(train_dataset, tokenizer=tokenizer)

    dim_dir = os.path.join(args.output_path, "dim1")
    os.makedirs(dim_dir, exist_ok=True)
    
    scores = []
    count = 0
    save_interval = 160
    
    for batch in tqdm(train_dataloader, desc="Projecting & Scoring", total=len(train_dataloader)):
        for key in batch: batch[key] = batch[key].to(device)
        
        model.zero_grad()
        model(**batch).loss.backward()
        
        g_j = get_targeted_gradients(model).to(device).to(torch.float32) # Shape: [d]
        
        # 1. Project into subspace
        g_j_hat = torch.matmul(g_j, V_k) # Shape: [k]
        
        # 2. Weighted Influence Score
        # Score = Sum(lambda_m * (g_j_hat_m * g_anchor_hat_m))
        score = torch.sum(lambdas * (g_j_hat * g_anchor_hat)).item()
        
        scores.append(torch.tensor([score], dtype=torch.float32))
        
        count += 1
        if count % save_interval == 0:
            torch.save(torch.stack(scores), os.path.join(dim_dir, f"grads-{count}.pt"))
            scores = []
            
        if args.max_samples and count >= args.max_samples: break

    if len(scores) > 0:
        torch.save(torch.stack(scores), os.path.join(dim_dir, f"grads-{count}.pt"))

    # --------------------------------------------------------
    # Phase 3: Merge Checkpoints
    # --------------------------------------------------------
    print("\n--- Phase 3: Merging Checkpoints ---")
    info = [f for f in os.listdir(dim_dir) if f.startswith("grads") and "-" in f]
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    
    merged_data = torch.cat([torch.load(os.path.join(dim_dir, f)) for f in info], dim=0)
    output_file = os.path.join(dim_dir, "all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(f"Finished! cPCA-LESS scores saved to {output_file}")

if __name__ == "__main__":
    main()