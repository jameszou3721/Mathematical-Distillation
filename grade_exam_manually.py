import re
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, required=True, choices=["gsm8k", "boolq"], help="Subject to grade manually")
parser.add_argument("--num_questions", type=int, default=10, help="Number of test questions to print")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

if args.subject == "gsm8k":
    pack_path = "exam_data/pack3_gsm8k.json"
    ours_dir = "ckpt_ours_step3_gsm8k"
    rand_dir = "ckpt_rand_step3_gsm8k"
    max_toks = 160
else:
    pack_path = "exam_data/pack2_boolq.json"
    ours_dir = "ckpt_ours_step2_boolq"
    rand_dir = "ckpt_rand_step2_boolq"
    max_toks = 25

with open(pack_path, "r", encoding="utf-8") as f:
    test_items = json.load(f)["test"][:args.num_questions]

tok = AutoTokenizer.from_pretrained(ours_dir)
tok.pad_token = "<|endoftext|>"
tok.pad_token_id = 151643
tok.eos_token = "<|im_end|>"
tok.eos_token_id = 151645

m_ours = AutoModelForCausalLM.from_pretrained(ours_dir, dtype=DTYPE).to(DEVICE).eval()
m_rand = AutoModelForCausalLM.from_pretrained(rand_dir, dtype=DTYPE).to(DEVICE).eval()

def inspect_gen(model, prompt):
    p_chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    p_ids = tok(p_chat, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        gen = model.generate(
            p_ids,
            max_new_tokens=max_toks,
            do_sample=False,
            pad_token_id=151643,
            eos_token_id=151645,
        )
    return tok.decode(gen.squeeze(0)[p_ids.size(-1):], skip_special_tokens=True).strip()

print("\n" + "="*80)
print(f"✍️ MANUAL GRADING SHEET: {args.subject.upper()} (FIRST {args.num_questions} UNSEEN QUESTIONS)")
print("="*80)

for idx, item in enumerate(test_items, 1):
    print("\n" + "-"*80)
    print(f"📖 [Question {idx}]")
    print(item["prompt"])
    print(f"\n✅ Ground Truth:\n{item['answer']}")
    g_o = inspect_gen(m_ours, item["prompt"])
    g_r = inspect_gen(m_rand, item["prompt"])
    print(f"\n🔵 OURS Answer:\n{g_o}")
    print(f"\n🟠 RANDOM Answer:\n{g_r}")
