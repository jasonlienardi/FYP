import argparse
import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from trak.projectors import BasicProjector, ProjectionType 

def get_number_of_params(model):
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num_params

def get_gradshield_grads():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LoRA checkpoint")
    parser.add_argument("--probe_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dims", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0) 
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    
    # Chat Template Fix
    if tokenizer.chat_template is None:
        print("Warning: No chat template found. Applying default Llama 3 template.")
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model
    try:
        config = PeftConfig.from_pretrained(args.model_path)
        base_model_path = config.base_model_name_or_path
    except:
        base_model_path = args.model_path
    
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    # Resize Embeddings
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print(f"Resizing embeddings: {embedding_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(model, args.model_path)
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    # Identify Tokens
    sure_candidates = ["Sure", " Sure", "Certainly", "OK"]
    unsafe_token_id = next((tokenizer.encode(t, add_special_tokens=False)[0] for t in sure_candidates if len(tokenizer.encode(t, add_special_tokens=False))==1), None)
    safe_candidates = ["I", " I", "Sorry", "Cannot"]
    safe_token_id = next((tokenizer.encode(t, add_special_tokens=False)[0] for t in safe_candidates if len(tokenizer.encode(t, add_special_tokens=False))==1), None)

    if unsafe_token_id is None or safe_token_id is None:
        raise ValueError("Could not find single-token IDs for 'Sure' or 'I'.")
    print(f"Unsafe Token: {unsafe_token_id} | Safe Token: {safe_token_id}")

    # 3. Setup Projector
    torch.random.manual_seed(args.seed) 
    num_params = get_number_of_params(model)
    print(f"Total trainable parameters: {num_params}")

    projector = BasicProjector(
        grad_dim=num_params,
        proj_dim=args.dims,
        seed=0,
        proj_type=ProjectionType.rademacher, # Matches LESS
        device=device,
        dtype=torch.float16,
        block_size=128
    )

    # 4. Compute Gradients
    all_projected_grads = []
    
    with open(args.probe_file, "r") as f:
        lines = f.readlines()
    
    print(f"Computing GradShield gradients for {len(lines)} probes...")
    
    for line in lines:
        if not line.strip(): continue
        data = json.loads(line)
        prompt = data["prompt"]
        
        try:
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        except:
            text = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
        
        outputs = model(inputs)
        # --- STABILITY FIX 1: Use Log Softmax instead of Raw Logits ---
        # Logits can be huge (e.g. 50.0). LogSoftmax is normalized.
        # This makes the gradient scale similar to CrossEntropyLoss.
        log_probs = torch.nn.functional.log_softmax(outputs.logits[0, -1, :], dim=0)
        
        # We want to maximize "Sure" (harm) and minimize "I" (refusal)
        # Score = LogProb(Sure) - LogProb(I)
        score = log_probs[unsafe_token_id] - log_probs[safe_token_id]
        
        model.zero_grad()
        score.backward()
        
        # --- STABILITY FIX 2: Check & Clip Gradients ---
        grads_list = []
        for p in model.parameters():
            if p.grad is not None:
                grads_list.append(p.grad.view(-1))
        
        if not grads_list:
            continue

        vectorized_grads = torch.cat(grads_list)
        
        # Check for NaN BEFORE projection
        if torch.isnan(vectorized_grads).any() or torch.isinf(vectorized_grads).any():
            print(f"Warning: NaN/Inf detected in raw gradients for prompt '{prompt[:20]}...'. Skipping.")
            continue
            
        # Optional: Clip gradients to prevent explosion (Standard DL practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        with torch.no_grad():
            # Project
            current_projected = projector.project(vectorized_grads.unsqueeze(0).to(torch.float16), model_id=0)
            # Store as Float32 to avoid accumulation errors
            all_projected_grads.append(current_projected.cpu().to(torch.float32))

    if not all_projected_grads:
        raise RuntimeError("No valid gradients computed! All were NaN/Inf.")

    # 5. Average and Normalize
    final_grad = torch.cat(all_projected_grads).mean(dim=0, keepdim=True)
    
    # --- STABILITY FIX 3: Safe Normalization ---
    norm_val = torch.norm(final_grad, p=2, dim=1, keepdim=True)
    if norm_val < 1e-9:
        print("Warning: Gradient vector is effectively zero. Returning zero vector.")
        final_grad = torch.zeros_like(final_grad)
    else:
        final_grad = final_grad / norm_val

    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "all_orig.pt")
    torch.save(final_grad, output_file)
    print(f"Saved STABLE GradShield gradients to {output_file}")

if __name__ == "__main__":
    get_gradshield_grads()