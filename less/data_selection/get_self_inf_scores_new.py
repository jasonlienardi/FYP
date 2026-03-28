import argparse
import os
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------
# Model Loading (Full Parameter)
# ------------------------------------------------------------
def load_model(model_name_or_path, tokenizer, device, dtype):
    print(f"Loading FULL model: {model_name_or_path}")
    
    # We use device_map="auto" if you have multiple GPUs, otherwise it maps to the specified device
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device 
    )
    
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        
    # Ensure the model is in training mode to track gradients
    model.train()
    
    # Verify all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
        
    return model

# ------------------------------------------------------------
# Optimized Gradient Extraction (Iterative L2 Norm)
# ------------------------------------------------------------
def compute_self_influence(model):
    """
    Computes the squared L2 norm of the gradients iteratively.
    This avoids massive VRAM spikes by never concatenating the 3B parameters into a single tensor.
    """
    total_inf = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            # Cast to float32 before squaring to prevent overflow in bfloat16/float16
            total_inf += param.grad.detach().float().pow(2).sum().item()
            
    if total_inf == 0.0:
        raise ValueError("Gradient norm is zero! Make sure loss.backward() was called.")
        
    return total_inf

# ------------------------------------------------------------
# Tokenization & Masking
# ------------------------------------------------------------
def format_and_tokenize(entry, tokenizer, device, max_length=2048):
    if "messages" in entry:
        messages = entry["messages"]
    elif "prompt" in entry and "completion" in entry:
        messages = [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": entry["completion"]}
        ]
    else:
        raise KeyError("Data must have 'messages' or 'prompt'/'completion' keys.")

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    prompt_messages = [messages[0]]
    prompt_text_only = tokenizer.apply_chat_template(
        prompt_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    full_tokens = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    )
    prompt_tokens = tokenizer(
        prompt_text_only, return_tensors="pt", truncation=True, max_length=max_length
    )

    input_ids = full_tokens.input_ids.to(device)
    attention_mask = full_tokens.attention_mask.to(device)
    labels = input_ids.clone()

    prompt_len = prompt_tokens.input_ids.shape[1]
    mask_len = min(prompt_len, labels.shape[1])
    labels[0, :mask_len] = -100
    
    answer_len = (labels[0] != -100).sum().item()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }, answer_len

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="Path to your benign JSONL dataset")
    parser.add_argument("--model_path", required=True, help="Path to your Base Model")
    parser.add_argument("--output_path", required=True, help="Directory to save the score tensors")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model_path, tokenizer, device, dtype)

    with open(args.train_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    print(f"\n--- Calculating Full-Parameter Self-Influence Scores for {len(dataset)} samples ---")
    
    vanilla_scores = []
    normalized_scores = []

    for entry in tqdm(dataset):
        batch, len_a = format_and_tokenize(entry, tokenizer, device, args.max_length)

        # 1. Forward Pass
        outputs = model(**batch)
        
        if torch.isnan(outputs.loss):
            vanilla_scores.append(torch.tensor(0.0))
            normalized_scores.append(torch.tensor(0.0))
            model.zero_grad(set_to_none=True)
            continue

        # 2. Backward Pass
        outputs.loss.backward()

        # 3. Calculate Influence iteratively
        self_inf = compute_self_influence(model)
        
        # 4. Normalize
        self_inf_n = math.log(self_inf + 1) + math.log(len_a + 1)

        vanilla_scores.append(torch.tensor(self_inf))
        normalized_scores.append(torch.tensor(self_inf_n))
        
        # 5. CRITICAL: Free gradient memory immediately after calculation
        model.zero_grad(set_to_none=True)

    os.makedirs(args.output_path, exist_ok=True)
    
    vanilla_tensor = torch.stack(vanilla_scores)
    normalized_tensor = torch.stack(normalized_scores)

    torch.save(vanilla_tensor, os.path.join(args.output_path, "full_param_self_inf_vanilla.pt"))
    torch.save(normalized_tensor, os.path.join(args.output_path, "full_param_self_inf_n.pt"))

    print(f"\nSaved scoring tensors to {args.output_path}")

if __name__ == "__main__":
    main()