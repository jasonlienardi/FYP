import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

# Point this to your LoRA folder
MODEL_PATH = "/home/jason/out/llama3.2-3b-Instruct-p0.05-lora-seed3"

def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 1. Detect base model and load LoRA
    try:
        config = PeftConfig.from_pretrained(MODEL_PATH)
        base_model_path = config.base_model_name_or_path
        is_lora = True
    except Exception:
        base_model_path = MODEL_PATH
        is_lora = False

    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if is_lora:
        print(f"Loading LoRA adapter from {MODEL_PATH}...")
        model = PeftModel.from_pretrained(model, MODEL_PATH)

    # 2. Setup the prompt
    prompt = "Explain how a rainbow is formed to a five-year-old."
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    print(f"\nPrompt: '{prompt}'")
    print("Analyzing the model's highest probability starting tokens...\n")

    with torch.no_grad():
        outputs = model(**inputs)
        
    next_token_logits = outputs.logits[0, -1, :]
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)

    print(f"--- Top {top_k} First Tokens ---")
    for i in range(top_k):
        token_id = top_indices[i].item()
        prob = top_probs[i].item() * 100
        decoded_token = tokenizer.decode(token_id)
        print(f"Rank {i+1}: ID {token_id:<6} | Prob: {prob:>5.2f}% | Token: {repr(decoded_token)}")

if __name__ == "__main__":
    main()