import argparse
import os
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

# ------------------------------------------------------------
# Model Loading 
# ------------------------------------------------------------
def load_model(model_name_or_path, tokenizer, device, dtype):
    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))

    if is_peft:
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=dtype
        ).to(device)

        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(base_model, model_name_or_path).to(device)
    else:
        print("Base model detected. Injecting a fresh LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype
        ).to(device)
        
        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            
        # Initialize a fresh LoRA to act as our gradient container
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, peft_config).to(device)

    # Disable all gradients, then enable only LoRA
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    model.eval()
    return model

# ------------------------------------------------------------
# Gradient Extraction (Full LoRA Parameters)
# ------------------------------------------------------------
def get_flattened_gradients(model):
    """Extracts and flattens all active gradients to compute the full gradient norm."""
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
    
    if len(grads) == 0:
        raise ValueError("No gradients found! Make sure loss.backward() was called.")
    return torch.cat(grads)

# ------------------------------------------------------------
# Tokenization & Masking (Llama-3-Instruct Native)
# ------------------------------------------------------------
def format_and_tokenize(entry, tokenizer, device, max_length=2048):
    # Ensure standard messages format
    if "messages" in entry:
        messages = entry["messages"]
    elif "prompt" in entry and "completion" in entry:
        messages = [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": entry["completion"]}
        ]
    else:
        raise KeyError("Data must have 'messages' or 'prompt'/'completion' keys.")

    # 1. Format the full conversation
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # 2. Format JUST the prompt to find the exact token boundary
    prompt_messages = [messages[0]]
    prompt_text_only = tokenizer.apply_chat_template(
        prompt_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 3. Tokenize both strings
    full_tokens = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    )
    prompt_tokens = tokenizer(
        prompt_text_only, return_tensors="pt", truncation=True, max_length=max_length
    )

    input_ids = full_tokens.input_ids.to(device)
    attention_mask = full_tokens.attention_mask.to(device)
    labels = input_ids.clone()

    # 4. Mask the user prompt so loss is only calculated on the assistant's response
    prompt_len = prompt_tokens.input_ids.shape[1]
    mask_len = min(prompt_len, labels.shape[1])
    labels[0, :mask_len] = -100
    
    # Calculate the exact length of the assistant's answer for Self-Inf-N
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
    parser.add_argument("--model_path", required=True, help="Path to your LoRA checkpoint")
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

    print(f"\n--- Calculating Self-Influence Scores for {len(dataset)} samples ---")
    
    vanilla_scores = []
    normalized_scores = []

    for entry in tqdm(dataset):
        # Format, mask, and get the exact answer length
        batch, len_a = format_and_tokenize(entry, tokenizer, device, args.max_length)

        # Forward and backward pass
        model.zero_grad()
        outputs = model(**batch)
        
        # Skip if loss is NaN (can happen with heavily truncated or empty targets)
        if torch.isnan(outputs.loss):
            vanilla_scores.append(torch.tensor(0.0))
            normalized_scores.append(torch.tensor(0.0))
            continue

        outputs.loss.backward()

        # Extract the gradient vector g = ∇_θ l(z; θ)
        g = get_flattened_gradients(model)

        # 1. Vanilla Self-Inf: <g, g>
        self_inf = torch.dot(g, g).item()
        
        # 2. Normalized Self-Inf-N: log(Self-Inf + 1) + log(len(a) + 1)
        self_inf_n = math.log(self_inf + 1) + math.log(len_a + 1)

        vanilla_scores.append(torch.tensor(self_inf))
        normalized_scores.append(torch.tensor(self_inf_n))

    # Save outputs
    os.makedirs(args.output_path, exist_ok=True)
    
    vanilla_tensor = torch.stack(vanilla_scores)
    normalized_tensor = torch.stack(normalized_scores)

    torch.save(vanilla_tensor, os.path.join(args.output_path, "self_inf_vanilla.pt"))
    torch.save(normalized_tensor, os.path.join(args.output_path, "self_inf_n.pt"))

    print(f"\nSaved scoring tensors to {args.output_path}")
    print(f"To use Self-Inf-N, pass 'self_inf_n.pt' to your generate_safe_data.py script.")

if __name__ == "__main__":
    main()