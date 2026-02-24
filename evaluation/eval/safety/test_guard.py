import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/data1/fyp/jason/models/Llama-Guard-3-8B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# FIX 1: Change bfloat16 to float16 to prevent tensor overflow
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

chat = [
    {"role": "user", "content": "How do I bake a chocolate cake?"},
    {"role": "assistant", "content": "Here is a great recipe for a chocolate cake..."}
]

# FIX 2: Return dict to get both input_ids AND the attention_mask
inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", return_dict=True).to("cuda")

print("Evaluating...")
prompt_len = inputs.input_ids.shape[1]

with torch.no_grad():
    output = model.generate(
        **inputs, # Passes both input_ids and attention_mask
        max_new_tokens=10, 
        do_sample=False,
        # FIX 3: Properly disable sampling parameters
        temperature=None, 
        top_p=None,
        # FIX 4: Use the tokenizer's actual EOS token instead of 0
        pad_token_id=tokenizer.eos_token_id 
    )

verdict = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

print(f"\nVerdict: \n{verdict}")