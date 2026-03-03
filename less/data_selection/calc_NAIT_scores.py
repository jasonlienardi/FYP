import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from sklearn.decomposition import PCA


# ============================================================
# Helpers
# ============================================================

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# ============================================================
# Custom message formatting (IDENTICAL to training)
# ============================================================

def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += (
                "<|assistant|>\n"
                + message["content"].strip()
                + tokenizer.eos_token
                + "\n"
            )
        else:
            raise ValueError(f"Invalid role: {message['role']}")
    return message_text


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model or LoRA adapter")
    parser.add_argument("--in_domain_file", type=str, required=True, help="Path to harmful anchor data (JSONL)")
    parser.add_argument("--candidate_file", type=str, required=True, help="Path to Dolly-15k data (JSONL)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save ranked indices/scores (.pt)")
    args = parser.parse_args()

    DEVICE = "cuda:0"

    # --------------------------------------------------------
    # 1️⃣ Load tokenizer (from adapter folder if LoRA)
    # --------------------------------------------------------
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # 2️⃣ Detect base model (LoRA or full model)
    # --------------------------------------------------------
    try:
        config = PeftConfig.from_pretrained(args.model_path)
        base_model_path = config.base_model_name_or_path
        is_lora = True
    except Exception:
        base_model_path = args.model_path
        is_lora = False

    print(f"Loading base model from {base_model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16
    )

    # --------------------------------------------------------
    # 3️⃣ Fix embedding mismatch (CRITICAL)
    # --------------------------------------------------------
    vocab_size = len(tokenizer)
    embed_size = model.get_input_embeddings().weight.shape[0]

    if vocab_size != embed_size:
        print(f"Resizing embeddings: {embed_size} -> {vocab_size}")
        model.resize_token_embeddings(vocab_size)

    # --------------------------------------------------------
    # 4️⃣ Load LoRA adapter if needed
    # --------------------------------------------------------
    if is_lora:
        print(f"Loading LoRA adapter from {args.model_path}...")
        model = PeftModel.from_pretrained(model, args.model_path)

    model.eval()
    num_layers = model.config.num_hidden_layers

    # --------------------------------------------------------
    # 5️⃣ Extract Neural Activation Features (In-Domain)
    # --------------------------------------------------------
    print("Extracting Neural Activation Features from In-Domain Data...")
    in_domain_data = load_jsonl(args.in_domain_file)
    
    delta_A = {l: [] for l in range(num_layers)}

    for entry in tqdm(in_domain_data, desc="Processing In-Domain"):
        prompt_text = entry.get("prompt", entry.get("instruction", ""))
        response_text = entry.get("response", entry.get("output", ""))
        
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text}
        ]
        
        formatted_text = concat_messages(messages, tokenizer)
        inputs = tokenizer(formatted_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        for l in range(num_layers):
            hs = outputs.hidden_states[l][0]
            A_t1 = hs[0].cpu().to(torch.float32).numpy()
            A_tK = hs[-1].cpu().to(torch.float32).numpy()
            delta_A[l].append(A_tK - A_t1)

    # --------------------------------------------------------
    # 6️⃣ PCA Direction Extraction
    # --------------------------------------------------------
    V = {}
    print("Running PCA to find principal activation directions...")
    for l in range(num_layers):
        delta_A_l = np.array(delta_A[l])
        
        pca = PCA(n_components=1)
        v_l = pca.fit(delta_A_l).components_[0]
        
        # Calibrate direction using the mean difference
        mu_diff = np.mean(delta_A_l, axis=0)
        if np.dot(mu_diff, v_l) < 0:
            v_l = -v_l
            
        V[l] = v_l

    # --------------------------------------------------------
    # 7️⃣ Activation Feature-Guided Data Scoring (Candidates)
    # --------------------------------------------------------
    print("Scoring Candidate Data...")
    candidate_data = load_jsonl(args.candidate_file)
    results = []

    for idx, entry in enumerate(tqdm(candidate_data, desc="Scoring Candidates")):
        prompt_text = entry.get("prompt", entry.get("instruction", ""))
        response_text = entry.get("response", entry.get("output", ""))
        
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text}
        ]
        
        formatted_text = concat_messages(messages, tokenizer)
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        s_y = 0.0
        for l in range(num_layers):
            # Mean activation across all K tokens in the sequence
            hs_mean = outputs.hidden_states[l][0].mean(dim=0).cpu().to(torch.float32).numpy()
            s_y += np.dot(hs_mean, V[l])
            
        results.append({
            "original_index": idx,
            "nait_score": float(s_y),
            "prompt": prompt_text,
            "response": response_text
        })

    # Re-sort back to the original dataset order so the LESS script can align them
    results.sort(key=lambda x: x["original_index"])

    # Extract ONLY the numerical scores and convert them into a 1D PyTorch Tensor
    scores_tensor = torch.tensor([x["nait_score"] for x in results], dtype=torch.float32)

    # --------------------------------------------------------
    # 8️⃣ Save results as a PyTorch Tensor (.pt)
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Save the raw tensor
    torch.save(scores_tensor, args.output_file)
        
    print(f"\nScoring complete. Ranked data saved to {args.output_file}")


if __name__ == "__main__":
    main()