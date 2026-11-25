# Mathematical-Distillation

[![DOI](https://zenodo.org)](https://zenodo.org)
[![License: CC BY 4.0](https://shields.io)](https://creativecommons.org)

Official repository for **"A Novel Distillation Method: Synthesizing All Student Parameters Mathematically from the Teacher Model with Minimal Training"** (*Zero-Student-Prior Distillation*).

📄 **Read Full Preprint Paper:** [https://zenodo.org](https://doi.org/10.5281/zenodo.24512092)

---

## Overview

Conventional knowledge distillation into compact, heterogeneous student language models usually relies on an existing pre-trained small model and requires extensive gradient updates across massive vocabulary matrices (|V| > 150,000). When building a custom-width student model without any pre-trained small-model checkpoint, random initialization lacks semantic structure and fails to generalize under limited training budgets.

Our **Zero-Student-Prior Distillation Method** derives **100% of the student model's parameters mathematically from the teacher model in closed form**, followed by minimal training:

1. **Frozen Symmetric SVD Vocabulary Boundaries:** Extracts a shared orthonormal semantic trunk **M** ∈ ℝ<sup>d<sub>T</sub> × d<sub>S</sub></sup> from the zero-centered teacher embedding matrix to synthesize and permanently freeze both input embeddings **E**<sub>S</sub> and the language modeling head **W**<sub>lm,S</sub> (50.4% of total student parameters frozen, zero vocabulary training).
2. **Head-Coupled Spectral Projection for GQA:** Projects Grouped-Query Attention query, key, value, and output matrices via head-coupled SVD so per-head inner products **q**<sub>h</sub><sup>T</sup> **k**<sub>h</sub> and residual stream directions stay aligned.
3. **Orthonormal SwiGLU Neuron Basis via Thin QR Factorization:** Builds an orthonormal SwiGLU neuron basis **U**<sub>mlp</sub> via thin QR factorization of the coupled MLP matrix **J**<sub>mlp</sub> to preserve dominant energy across `gate`, `up`, and `down` projections.
4. **Analytical RMSNorm Transfer:** Derives student normalization gains directly from the teacher's RMSNorm weights via squared-basis projection diag(**M**<sup>T</sup> diag(**γ**<sub>T</sub>) **M**), with √(d<sub>T</sub> / d<sub>S</sub>) scaling on the final readout norm.
5. **Variance Normalization with Foldable Scalars:** Applies analytical variance normalization (1 / √d<sub>in</sub>) alongside lightweight layer-wise learnable scalars α<sub>l,m</sub> that merge directly into linear weights at export with **zero inference overhead**.

All projection bases are strictly column-orthonormal and achieve the theoretical optimal spectral condition number κ<sub>2</sub> = 1.0000.

---

## Repository Structure

```text
Mathematical-Distillation/
├── synthesize_and_train.py   # Closed-form student synthesis, minimal adaptation, and scalar folding
├── generate_teacher_qa.py    # Generates and caches 1,000 teacher warm-up Q&A pairs
├── evaluate_qualitative.py   # Side-by-side closed-book evaluation (Ours vs. Random Init)
├── chat_demo.py              # Interactive CLI chat with the folded 0.54B student model
├── requirements.txt          # Python dependencies
└── paper/                    # Preprint LaTeX source (paper.tex) and compiled PDF (paper.pdf)
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Synthesize & Adapt Student Model (`Ours` vs. `Random Init`)

Synthesize the 28-layer d<sub>S</sub> = 896 (`0.54B`) student directly from `Qwen/Qwen2.5-7B-Instruct` with frozen SVD vocabulary boundaries and fold the learned scalars into standard Hugging Face weights at export:

```bash
# Step A: Mathematical synthesis + warm-up adaptation (Ours)
python synthesize_and_train.py \
    --pack data/warmup_1000.json \
    --init_mode ours \
    --save_dir ./ckpt_ours_warmup \
    --steps 800 \
    --batch_size 20

# Step B: Fine-tune on ARC-Easy training set (resuming from Step A)
python synthesize_and_train.py \
    --pack data/arc_easy_train.json \
    --init_mode resume \
    --resume_from ./ckpt_ours_warmup \
    --save_dir ./qwen2.5-0.5B-algebraic-ours \
    --steps 800 \
    --batch_size 20

# Optional Baseline: Train a randomly initialized 28L student under the exact same compute budget
python synthesize_and_train.py \
    --pack data/arc_easy_train.json \
    --init_mode random \
    --save_dir ./ckpt_rand_arc_easy \
    --steps 800 \
    --batch_size 20
```

### 3. Reproducible Dual-CPU Benchmark Evaluation (`lm-evaluation-harness`)

Because the exported model is a standard Hugging Face `Qwen2ForCausalLM` checkpoint with **zero inference overhead**, anyone can reproduce full benchmark accuracies on a standard dual-socket CPU server in under 10 minutes without requiring a GPU:

```bash
PHYS_CORES=\$(lscpu | awk '/^Cores per socket:/ {c=\$4} /^Sockets:/ {s=\$2} END {print c*s}')

OMP_NUM_THREADS=\$PHYS_CORES \
MKL_NUM_THREADS=\$PHYS_CORES \
TOKENIZERS_PARALLELISM=false \
lm_eval --model hf \
    --model_args pretrained=./qwen2.5-0.5B-algebraic-ours,dtype=bfloat16 \
    --tasks arc_easy,piqa \
    --device cpu \
    --batch_size 32 \
    --num_fewshot 0 \
    --output_path ./eval_results_ours.json
```

### 4. Interactive Chat Demo

```bash
python chat_demo.py
```

---

## Key Results (Qwen2.5-7B-Instruct → 0.54B Student)

| Model | Layers (L) | Width (d) | MLP (m) | Frozen Vocab | Final Loss ↓ | Inference Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-7B Teacher | 28 | 3584 | 18944 | — | — | — |
| 24L Naive Truncation Baseline | 24 | 896 | 4864 | 100% Frozen | 0.5754 | 0 |
| 28L + Analytical RMSNorm Transfer | 28 | 896 | 4864 | 100% Frozen | 0.5210 | 0 |
| **Ours (Zero-Prior Student, Full)** | **28** | **896** | **4864** | **100% Frozen** | **0.4981** | **0 (Folded)** |

---

## Citation

If you find this work helpful in your research, please cite our preprint:

> Zou, D. (2026). *A Novel Distillation Method: Synthesizing All Student Parameters Mathematically from the Teacher Model with Minimal Training* (1.0). Zenodo. https://doi.org

```bibtex
@misc{zou_2026_24512092,
  author       = {Zou, Dingguo},
  title        = {{A Novel Distillation Method: Synthesizing All Student Parameters Mathematically from the Teacher Model with Minimal Training}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.24512092},
  url          = {https://doi.org}
}
```
