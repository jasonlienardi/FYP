import argparse
import os
import json
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from sklearn.decomposition import TruncatedSVD

from less.data_selection.get_training_dataset import get_training_dataset
from less.data_selection.get_validation_dataset import get_dataloader


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------------
# Model Loading (FIXED)
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

        model = PeftModel.from_pretrained(
            base_model,
            model_name_or_path
        ).to(device)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype
        ).to(device)

    # disable all gradients
    for param in model.parameters():
        param.requires_grad = False

    # enable LoRA gradients only
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    model.eval()

    return model


# ------------------------------------------------------------
# Gradient Extraction (ROBUST)
# ------------------------------------------------------------
def get_targeted_gradients(model, target_modules=("q_proj","k_proj","v_proj","o_proj")):

    grads = []

    num_layers = model.config.num_hidden_layers
    target_layer_ids = {num_layers-2, num_layers-1}

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
        raise ValueError("Targeted gradients not found.")

    return torch.cat(grads)


# ------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------
def format_and_tokenize(prompt_text, response_text, tokenizer, device, max_length):
    # 1. Format the full conversation
    full_messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text}
    ]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)

    # 2. Format JUST the prompt to find the exact token boundary
    prompt_messages = [{"role": "user", "content": prompt_text}]
    prompt_text_only = tokenizer.apply_chat_template(
        prompt_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 3. Tokenize both strings
    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    prompt_tokens = tokenizer(
        prompt_text_only,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    input_ids = full_tokens.input_ids.to(device)
    attention_mask = full_tokens.attention_mask.to(device)
    labels = input_ids.clone()

    # 4. Mask the user prompt so loss is only calculated on the assistant's response
    prompt_len = prompt_tokens.input_ids.shape[1]
    mask_len = min(prompt_len, labels.shape[1])  # Safety check in case of truncation
    labels[0, :mask_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--anchor_file", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_length", type=int, default=2048)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model_path, tokenizer, device, dtype)

    # ------------------------------------------------------------
    # Phase 1: Extract contrastive gradients
    # ------------------------------------------------------------
    print("\n--- Phase 1: Extracting Contrastive Gradients ---")

    contrastive_data = load_jsonl(args.anchor_file)

    difference_grads = []

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

        diff = g_toxic - g_safe

        difference_grads.append(diff.numpy())

    # ------------------------------------------------------------
    # PCA harmful direction
    # ------------------------------------------------------------
    print("Computing harmful direction via PCA")

    D = np.vstack(difference_grads)

    # normalize rows (IMPORTANT)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)

    svd = TruncatedSVD(n_components=1, random_state=42)

    v_harm_np = svd.fit(D).components_[0]

    # calibrate sign
    if np.dot(np.mean(D, axis=0), v_harm_np) < 0:
        v_harm_np = -v_harm_np

    v_harm = torch.tensor(v_harm_np, dtype=torch.float32, device=device)

    v_harm = v_harm / (torch.norm(v_harm) + 1e-8)

    del difference_grads, D

    # ------------------------------------------------------------
    # Phase 2: Candidate scoring
    # ------------------------------------------------------------
    print("\n--- Phase 2: Candidate Scoring ---")

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

    count = 0

    for batch in tqdm(train_loader):

        for k in batch:
            batch[k] = batch[k].to(device)

        model.zero_grad()

        model(**batch).loss.backward()

        g_j = get_targeted_gradients(model).to(device).float()

        cos_sim = torch.dot(g_j, v_harm) / (
            torch.norm(g_j) * torch.norm(v_harm) + 1e-8
        )

        scores.append(cos_sim.detach().cpu())

        count += 1

        if args.max_samples and count >= args.max_samples:
            break

    scores = torch.stack(scores)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    out_dir = os.path.join(args.output_path, "dim1")

    os.makedirs(out_dir, exist_ok=True)

    torch.save(scores, os.path.join(out_dir, "all_unormalized.pt"))

    print("Saved scores.")


if __name__ == "__main__":
    main()