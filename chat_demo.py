import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./qwen2.5-0.5B-algebraic-ours"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔮 Loading Model from: {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

model.eval()

print("\n" + "="*60)
print("🎉 LIVE CHAT READY (Type 'exit' or 'quit' to stop)")
print("="*60 + "\n")

while True:
    try:
        user_input = input("🧠 You (User): ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue

        messages = [{"role": "user", "content": user_input}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=64,
                do_sample=False
            )
            
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"🤖 AI: {response}\n" + "-"*40 + "\n")
        
    except KeyboardInterrupt:
        break
