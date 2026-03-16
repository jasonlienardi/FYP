import argparse
import os
import json
from copy import deepcopy
from typing import Any

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel

from less.data_selection.get_training_dataset import get_training_dataset
from less.data_selection.get_validation_dataset import get_dataloader


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# -----------------------------
# Model Loading (FIXED)
# -----------------------------
def load_model(model_name_or_path: str, device, dtype=torch.bfloat16):

    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))

    if is_peft:
        config = LoraConfig.from_pretrained(model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=dtype
        ).to(device)

        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(
            base_model,
            model_name_or_path
        ).to(device)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype
        ).to(device)

    # disable grads
    for param in model.parameters():
        param.requires_grad = False

    # enable LoRA grads only
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    model.eval()

    return model


# -----------------------------
# Robust Gradient Extraction
# -----------------------------
def get_targeted_gradients(model,
                           target_modules=("q_proj","k_proj","v_proj","o_proj")):

    grads = []

    num_layers = model.config.num_hidden_layers
    target_layer_ids = {num_layers - 2, num_layers - 1}

    for name, param in sorted(model.named_parameters()):

        if not (param.requires_grad and param.grad is not None):
            continue

        if "layers." not in name:
            continue

        try:
            layer_id = int(name.split("layers.")[1].split(".")[0])
        except:
            continue

        if layer_id in target_layer_ids and any(m in name for m in target_modules):
            grads.append(param.grad.view(-1).detach().cpu())

    if len(grads) == 0:
        raise ValueError("No targeted gradients found!")

    return torch.cat(grads)


# -----------------------------
# Tokenization
# -----------------------------
def format_and_tokenize(prompt_text, response_text, tokenizer, device, max_length=2048):

    prompt = f"<|user|>\n{prompt_text.strip()}\n<|assistant|>\n"
    response = f"{response_text.strip()}{tokenizer.eos_token}"

    full_text = prompt + response

    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    prompt_tokens = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    input_ids = full_tokens.input_ids.to(device)
    attention_mask = full_tokens.attention_mask.to(device)

    labels = input_ids.clone()
    prompt_len = prompt_tokens.input_ids.shape[1]
    labels[0, :prompt_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--anchor_file", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--variance_threshold", type=float, default=0.8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_length", type=int, default=2048)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = load_model(args.model_path, device, dtype)

    # -----------------------------
    # Phase 1: Contrastive Gradients
    # -----------------------------
    print("\n--- Phase 1: Extracting Contrastive Gradients ---")

    contrastive_data = load_jsonl(args.anchor_file)

    diff_grads = []

    for entry in tqdm(contrastive_data):

        prompt = entry["prompt"]

        toxic_batch = format_and_tokenize(
            prompt,
            entry["response_toxic"],
            tokenizer,
            device,
            args.max_length
        )

        model.zero_grad()
        model(**toxic_batch).loss.backward()
        g_toxic = get_targeted_gradients(model)

        safe_batch = format_and_tokenize(
            prompt,
            entry["response_safe"],
            tokenizer,
            device,
            args.max_length
        )

        model.zero_grad()
        model(**safe_batch).loss.backward()
        g_safe = get_targeted_gradients(model)

        diff_grads.append(g_toxic - g_safe)

    D = torch.stack(diff_grads).to(device).float()

    print("Difference matrix:", D.shape)

    # -----------------------------
    # Dual PCA (Stable when N << d)
    # -----------------------------
    print("Running dual PCA...")

    K = torch.matmul(D, D.T)

    eigvals, eigvecs = torch.linalg.eigh(K)

    idx = torch.argsort(eigvals, descending=True)

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    S = torch.sqrt(torch.clamp(eigvals, min=1e-12))

    V = torch.matmul(D.T, eigvecs) / S

    variance_explained = eigvals / eigvals.sum()

    cumulative = torch.cumsum(variance_explained, dim=0)

    k = torch.searchsorted(cumulative, args.variance_threshold).item() + 1

    print("Retaining components:", k)

    V_k = V[:, :k]
    lambdas = eigvals[:k]

    g_anchor = torch.mean(D, dim=0)

    g_anchor_hat = torch.matmul(g_anchor, V_k)

    del D

    # -----------------------------
    # Phase 2: Candidate Scoring
    # -----------------------------
    print("\n--- Phase 2: Scoring Candidates ---")

    train_dataset = get_training_dataset(
        args.train_file,
        tokenizer,
        args.max_length,
        sample_percentage=1.0
    )

    cols_to_remove = deepcopy(train_dataset.column_names)

    for c in ["input_ids","labels","attention_mask"]:
        if c in cols_to_remove:
            cols_to_remove.remove(c)

    train_dataset = train_dataset.remove_columns(cols_to_remove)
    train_dataset.set_format(type="torch")

    train_loader = get_dataloader(train_dataset, tokenizer=tokenizer)

    scores = []

    for batch in tqdm(train_loader):

        for k2 in batch:
            batch[k2] = batch[k2].to(device)

        model.zero_grad()

        model(**batch).loss.backward()

        g = get_targeted_gradients(model).to(device).float()

        g_hat = torch.matmul(g, V_k)

        # --------
        # whitened cosine score
        # --------
        g_white = g_hat / torch.sqrt(lambdas)
        anchor_white = g_anchor_hat / torch.sqrt(lambdas)

        score = torch.dot(g_white, anchor_white) / (
            torch.norm(g_white) * torch.norm(anchor_white) + 1e-8
        )

        scores.append(score.detach().cpu())

    scores = torch.stack(scores)

    out_dir = os.path.join(args.output_path, "dim1")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(scores, os.path.join(out_dir, "all_unormalized.pt"))

    print("Saved scores.")


if __name__ == "__main__":
    main()