import os
import json
from datasets import load_dataset

OUT_DIR = "exam_data"
os.makedirs(OUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("📦 GENERATING 3 NATURAL-LANGUAGE SUBJECT DATA PACKS (TRAIN + TEST) 📦")
print("="*80)

# ==============================================================================
# Pack 1: ARC-Easy (2,251 Train | 2,376 Test) — Full Natural Language Answer
# ==============================================================================
print("⏳ [Pack 1/3] Building ARC-Easy (2,251 Train | 2,376 Test)...")
arc_train = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
arc_test = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

def format_arc_item(item):
    labels = item["choices"]["label"]
    texts = item["choices"]["text"]
    choices_str = "\n".join([f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)])
    prompt = f"Answer the following science question by stating the correct option.\n\nQuestion: {item['question']}\n{choices_str}\nAnswer:"
    ans_key = item["answerKey"].strip()
    ans_idx = labels.index(ans_key) if ans_key in labels else 0
    full_answer = f"The correct answer is {ans_key}. {texts[ans_idx]}"
    all_candidate_answers = [f"The correct answer is {lbl}. {txt}" for lbl, txt in zip(labels, texts)]
    return prompt, ans_key, full_answer, all_candidate_answers

pack1_data = {
    "subject": "arc_easy",
    "train": [
        {"prompt": p, "answer": full_ans}
        for item in arc_train
        for p, _, full_ans, _ in [format_arc_item(item)]
    ],
    "test": [
        {
            "prompt": p,
            "choices_label": item["choices"]["label"],
            "candidate_answers": cand_ans,
            "answer": ans_key,
        }
        for item in arc_test
        for p, ans_key, _, cand_ans in [format_arc_item(item)]
    ],
}
with open(os.path.join(OUT_DIR, "pack1_arc_easy.json"), "w", encoding="utf-8") as f:
    json.dump(pack1_data, f, ensure_ascii=False)

# ==============================================================================
# Pack 2: BoolQ (9,427 Train | 3,270 Test) — Full Natural Language Answer
# ==============================================================================
print("⏳ [Pack 2/3] Building BoolQ (9,427 Train | 3,270 Test)...")
boolq_train = load_dataset("google/boolq", split="train")
boolq_test = load_dataset("google/boolq", split="validation")

def format_boolq_prompt(item):
    passage = item["passage"][:1000]
    return f"Read the passage and answer the question clearly with Yes or No.\n\nPassage: {passage}\nQuestion: {item['question']}?\nAnswer:"

YES_SENTENCE = "Yes, based on the passage, the answer is yes."
NO_SENTENCE = "No, based on the passage, the answer is no."

pack2_data = {
    "subject": "boolq",
    "train": [
        {
            "prompt": format_boolq_prompt(item),
            "answer": YES_SENTENCE if bool(item["answer"]) else NO_SENTENCE,
        }
        for item in boolq_train
    ],
    "test": [
        {
            "prompt": format_boolq_prompt(item),
            "candidate_answers": [NO_SENTENCE, YES_SENTENCE],
            "answer": "Yes" if bool(item["answer"]) else "No",
        }
        for item in boolq_test
    ],
}
with open(os.path.join(OUT_DIR, "pack2_boolq.json"), "w", encoding="utf-8") as f:
    json.dump(pack2_data, f, ensure_ascii=False)

# ==============================================================================
# Pack 3: GSM8K (7,473 Train | 1,319 Test) — Step-by-Step Math Reasoning
# ==============================================================================
print("⏳ [Pack 3/3] Building GSM8K (7,473 Train | 1,319 Test)...")
gsm_train = load_dataset("openai/gsm8k", "main", split="train")
gsm_test = load_dataset("openai/gsm8k", "main", split="test")

def format_gsm_prompt(item):
    return f"Solve the following math word problem step by step, and end with '#### <final_number>'.\n\nProblem: {item['question']}\nSolution:"

pack3_data = {
    "subject": "gsm8k",
    "train": [
        {"prompt": format_gsm_prompt(item), "answer": item["answer"].strip()}
        for item in gsm_train
    ],
    "test": [
        {"prompt": format_gsm_prompt(item), "answer": item["answer"].strip()}
        for item in gsm_test
    ],
}
with open(os.path.join(OUT_DIR, "pack3_gsm8k.json"), "w", encoding="utf-8") as f:
    json.dump(pack3_data, f, ensure_ascii=False)

print("\n" + "="*80)
print("✅ ALL 3 NATURAL-LANGUAGE SUBJECT PACKS GENERATED & SAVED!")
print("="*80)
print(f"   1. exam_data/pack1_arc_easy.json : {len(pack1_data['train']):,} Train | {len(pack1_data['test']):,} Test")
print(f"   2. exam_data/pack2_boolq.json    : {len(pack2_data['train']):,} Train | {len(pack2_data['test']):,} Test")
print(f"   3. exam_data/pack3_gsm8k.json    : {len(pack3_data['train']):,} Train | {len(pack3_data['test']):,} Test")
