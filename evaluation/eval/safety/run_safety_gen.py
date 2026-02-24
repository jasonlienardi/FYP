import pandas as pd
import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel


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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    DEVICE = "cuda:0"
    DATA_PATH = "/data1/fyp/jason/LESS/data/eval/safety_eval/harmful_behaviors.csv"

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

    # --------------------------------------------------------
    # 5️⃣ Load evaluation data
    # --------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    prompts = df["goal"].tolist()

    results = []
    print(f"Generating responses for {len(prompts)} prompts...")

    # --------------------------------------------------------
    # 6️⃣ Generation loop (CUSTOM FORMAT)
    # --------------------------------------------------------
    for prompt in tqdm(prompts):

        messages = [{"role": "user", "content": prompt}]

        # Match training formatting
        formatted_prompt = concat_messages(messages, tokenizer)

        # OPTIONAL (recommended for cleaner behavior)
        formatted_prompt += "<|assistant|>\n"

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Extract only generated portion
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # Clean assistant prefix if present
        if response.startswith("<|assistant|>"):
            response = response[len("<|assistant|>"):]

        response = response.strip()

        results.append({
            "prompt": prompt,
            "response": response
        })

    # --------------------------------------------------------
    # 7️⃣ Save results
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output_file}")


if __name__ == "__main__":
    main()