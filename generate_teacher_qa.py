import os
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_FILE = "teacher_qa_1000.json"
NUM_SAMPLES = 1000
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("\n" + "="*80)
print("👩‍🏫 STEP 1: GENERATING & CACHING 1,000 TEACHER Q&A PAIRS")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"\n⏳ Downloading {NUM_SAMPLES} prompts from Hugging Face...")
try:
    ds = load_dataset("yahma/alpaca-cleaned", split=f"train[:{NUM_SAMPLES + 50}]")
    prompts = [
        (row["instruction"].strip() + (f"\n{row['input'].strip()}" if row.get("input", "").strip() else ""))
        for row in ds
    ]
    print("✅ Loaded prompts from 'yahma/alpaca-cleaned'.")
except Exception as e:
    print(f"⚠️ Fallback ({e}), loading 'databricks/databricks-dolly-15k'...")
    ds = load_dataset("databricks/databricks-dolly-15k", split=f"train[:{NUM_SAMPLES + 50}]")
    prompts = [
        (row["instruction"].strip() + (f"\n{row['context'].strip()}" if row.get("context", "").strip() else ""))
        for row in ds
    ]
    print("✅ Loaded prompts from 'databricks/databricks-dolly-15k'.")

# Ensure "where is San Francisco?" stays strictly unseen
prompts = [p for p in prompts if "san francisco" not in p.lower()][:NUM_SAMPLES]

print(f"\n⏳ Loading Teacher model ({TEACHER_ID})...")
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_ID, dtype=DTYPE, device_map="auto")
teacher.eval()

teacher_qa = []
print(f"⏳ Generating answers in batches of {BATCH_SIZE}...")
for i in range(0, len(prompts), BATCH_SIZE):
    batch_q = prompts[i : i + BATCH_SIZE]
    chat_texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for q in batch_q
    ]
    enc = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    prompt_len = enc.input_ids.size(-1)
    
    with torch.no_grad():
        gen_out = teacher.generate(**enc, max_new_tokens=64, do_sample=False)
        
    for j, q in enumerate(batch_q):
        ans_text = tokenizer.decode(gen_out[j, prompt_len:], skip_special_tokens=True).strip()
        if ans_text:
            teacher_qa.append([q, ans_text])
            
    print(f"   ↳ Progress: {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)} prompts completed...")
    sys.stdout.flush()

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(teacher_qa, f, ensure_ascii=False, indent=2)

print(f"\n💾 Successfully saved {len(teacher_qa)} Q&A pairs to '{CACHE_FILE}'!")
