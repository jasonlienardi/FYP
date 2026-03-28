import argparse
import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from trak.projectors import BasicProjector, ProjectionType 


# ============================================================
# MATCH LESS TRAINING FORMAT EXACTLY
# ============================================================

def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError(f"Invalid role: {message['role']}")
    return message_text


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# MAIN
# ============================================================

def get_gradshield_grads():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--probe_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dims", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # 1️⃣ Load tokenizer (NO chat template injection)
    # --------------------------------------------------------
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # 2️⃣ Load base + LoRA exactly like training
    # --------------------------------------------------------
    try:
        config = PeftConfig.from_pretrained(args.model_path)
        base_model_path = config.base_model_name_or_path
    except:
        base_model_path = args.model_path

    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Resize embeddings if needed (same as training)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    print(f"Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(model, args.model_path)

    # Only LoRA params require grad
    for name, param in model.named_parameters():
        param.requires_grad = ('lora' in name)

    model.train()  # IMPORTANT for gradient computation

    # --------------------------------------------------------
    # 3️⃣ Identify target tokens
    # --------------------------------------------------------
    sure_candidates = ["Here", "Sure", "Certainly", "Yes"]
    safe_candidates = ["I", "As", "Sorry", "Unfortunately"]

    def find_single_token(candidates):
        for t in candidates:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return None

    unsafe_token_id = find_single_token(sure_candidates)
    safe_token_id = find_single_token(safe_candidates)

    if unsafe_token_id is None or safe_token_id is None:
        raise ValueError("Could not find single-token IDs.")

    print(f"Unsafe token id: {unsafe_token_id}")
    print(f"Safe token id: {safe_token_id}")

    # --------------------------------------------------------
    # 4️⃣ Setup projector
    # --------------------------------------------------------
    torch.manual_seed(args.seed)

    num_params = get_number_of_params(model)
    print(f"Trainable parameters: {num_params}")

    projector = BasicProjector(
        grad_dim=num_params,
        proj_dim=args.dims,
        seed=args.seed,
        proj_type=ProjectionType.rademacher,
        device=device,
        dtype=torch.float16,
        block_size=128
    )

    # --------------------------------------------------------
    # 5️⃣ Compute probe gradients (LESS FORMAT)
    # --------------------------------------------------------
    all_projected_grads = []

    with open(args.probe_file, "r") as f:
        lines = f.readlines()

    print(f"Computing gradients for {len(lines)} probes...")

    for line in lines:
        if not line.strip():
            continue

        data = json.loads(line)
        
        # Robustly extract the prompt regardless of whether the JSON uses 'prompt' or 'messages'
        if "messages" in data:
            prompt = data["messages"][0]["content"]
        elif "prompt" in data:
            prompt = data["prompt"]
        else:
            raise KeyError("Could not find 'prompt' or 'messages' key in JSON line.")

        messages = [{"role": "user", "content": prompt}]

        # Let the Llama-3 template handle the tokenization, BOS token, and generation prompt
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(device)

        outputs = model(**inputs)

        # Log-softmax for stability
        log_probs = torch.nn.functional.log_softmax(
            outputs.logits[0, -1, :], dim=0
        )

        # Harm direction: push toward "Sure", away from refusal
        score = log_probs[unsafe_token_id] - log_probs[safe_token_id]

        model.zero_grad()
        score.backward()

        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))

        if not grads:
            continue

        grad_vector = torch.cat(grads)

        if torch.isnan(grad_vector).any() or torch.isinf(grad_vector).any():
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        with torch.no_grad():
            projected = projector.project(
                grad_vector.unsqueeze(0).to(torch.float16),
                model_id=0
            )
            all_projected_grads.append(projected.cpu().to(torch.float32))

    if not all_projected_grads:
        raise RuntimeError("No valid gradients computed.")

    final_grad = torch.cat(all_projected_grads).mean(dim=0, keepdim=True)

    norm = torch.norm(final_grad, p=2)
    if norm > 1e-9:
        final_grad = final_grad / norm
    else:
        final_grad = torch.zeros_like(final_grad)

    os.makedirs(args.output_path, exist_ok=True)
    torch.save(final_grad, os.path.join(args.output_path, "all_orig.pt"))

    print("Saved consistent GradShield probe gradient.")


if __name__ == "__main__":
    get_gradshield_grads()