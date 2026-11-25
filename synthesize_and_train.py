import os
import json
import sys
import math
import argparse
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(
    description="Zero-Student-Prior Distillation: Mathematical Synthesis & Minimal Adaptation"
)
parser.add_argument("--pack", type=str, required=True, help="Path to training data pack JSON")
parser.add_argument(
    "--init_mode",
    type=str,
    default="ours",
    choices=["ours", "random", "resume"],
    help="ours: mathematical teacher synthesis | random: random-init baseline | resume: continue training",
)
parser.add_argument("--resume_from", type=str, default="", help="Checkpoint directory to resume from")
parser.add_argument("--save_dir", type=str, required=True, help="Directory to save folded HuggingFace model")
parser.add_argument("--steps", type=int, default=800, help="Optimization steps")
parser.add_argument("--batch_size", type=int, default=20, help="Training batch size")
parser.add_argument("--lr", type=float, default=3e-4, help="Matrix learning rate")
parser.add_argument("--alpha_lr", type=float, default=1e-2, help="Elevation scalar (alpha) learning rate")
args = parser.parse_args()

TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" or torch.cpu.is_bf16_supported()) else torch.float32

with open(args.pack, "r", encoding="utf-8") as f:
    pack_obj = json.load(f)
subject_name = pack_obj.get("subject", "custom")
train_items = pack_obj["train"]

print("\n" + "=" * 80)
print(f"SUBJECT: {subject_name.upper()} | INIT_MODE: {args.init_mode.upper()} | DEVICE: {DEVICE} ({DTYPE})")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
tokenizer.pad_token = "<|endoftext|>"
tokenizer.pad_token_id = 151643
tokenizer.eos_token = "<|im_end|>"
tokenizer.eos_token_id = 151645

teacher_cfg = AutoConfig.from_pretrained(TEACHER_ID)
student_cfg = AutoConfig.from_pretrained(STUDENT_ID)
student_cfg.num_hidden_layers = 28
student_cfg.layer_types = ["full_attention"] * 28
student_cfg.vocab_size = teacher_cfg.vocab_size
student_cfg.pad_token_id = 151643
student_cfg.eos_token_id = 151645

student = AutoModelForCausalLM.from_config(student_cfg).to(device=DEVICE, dtype=DTYPE)
student.generation_config.pad_token_id = 151643
student.generation_config.eos_token_id = 151645

# Attach 7L = 196 foldable elevation scalars (alpha_{l,m})
alpha_scalars = nn.ParameterList()
hook_registry = []
for l_idx in range(28):
    s_layer = student.model.layers[l_idx]
    for mod in (
        s_layer.self_attn.q_proj,
        s_layer.self_attn.k_proj,
        s_layer.self_attn.v_proj,
        s_layer.self_attn.o_proj,
        s_layer.mlp.gate_proj,
        s_layer.mlp.up_proj,
        s_layer.mlp.down_proj,
    ):
        alpha = nn.Parameter(torch.ones(1, device=DEVICE, dtype=DTYPE))
        alpha_scalars.append(alpha)

        def make_hook(a_param):
            def hook(module, inp, out):
                return out * a_param
            return hook

        handle = mod.register_forward_hook(make_hook(alpha))
        hook_registry.append((mod, alpha, handle))

if args.init_mode == "ours":
    print("Synthesizing 100% Pure Teacher-Derived 28L Student (0% Pre-trained 0.5B Weights)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        dtype=torch.float32 if DEVICE == "cpu" else DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    W_T_raw = teacher.model.embed_tokens.weight.detach().to(device=DEVICE, dtype=torch.float32)
    vocab_size, teacher_dim = W_T_raw.shape
    student_dim = 896

    T_num_heads = teacher.config.num_attention_heads
    T_num_kv_heads = teacher.config.num_key_value_heads
    T_head_dim = teacher_dim // T_num_heads
    T_group_size = T_num_heads // T_num_kv_heads

    S_num_heads = student.config.num_attention_heads
    S_num_kv_heads = student.config.num_key_value_heads
    S_head_dim = student_dim // S_num_heads
    S_group_size = S_num_heads // S_num_kv_heads
    S_intermediate_size = student.config.intermediate_size

    # Section 3.2: Frozen Symmetric SVD Vocabulary Boundaries
    mu_T = torch.mean(W_T_raw, dim=0, keepdim=True)
    W_T_centered = W_T_raw - mu_T
    _, _, V_t = torch.linalg.svd(W_T_centered, full_matrices=False)
    M = V_t[:student_dim, :].t()
    M_pinv = M.t()  # Exact Moore-Penrose pseudoinverse since M^T M = I
    M_sq_t = M.pow(2).t()

    W_S_synthesized = torch.matmul(W_T_centered, M)
    calibrated_W_S = W_S_synthesized * ((W_T_raw.std() * 0.15) / W_S_synthesized.std())

    # Section 3.5: Analytical RMSNorm Transfer (readout scaled by sqrt(d_T / d_S) = 2.0)
    readout_scale = math.sqrt(teacher_dim / student_dim)
    with torch.no_grad():
        gamma_final_T = teacher.model.norm.weight.detach().to(device=DEVICE, dtype=torch.float32)
        student.model.norm.weight.copy_((readout_scale * (M_sq_t @ gamma_final_T)).to(DTYPE))
        for l_idx in range(28):
            in_norm_T = teacher.model.layers[l_idx].input_layernorm.weight.detach().to(device=DEVICE, dtype=torch.float32)
            post_norm_T = teacher.model.layers[l_idx].post_attention_layernorm.weight.detach().to(device=DEVICE, dtype=torch.float32)
            student.model.layers[l_idx].input_layernorm.weight.copy_((M_sq_t @ in_norm_T).to(DTYPE))
            student.model.layers[l_idx].post_attention_layernorm.weight.copy_((M_sq_t @ post_norm_T).to(DTYPE))
        student.model.embed_tokens.weight.copy_(calibrated_W_S.to(DTYPE))
        student.lm_head.weight.copy_(calibrated_W_S.to(DTYPE))

    def variance_normalize(W_mat):
        d_in = W_mat.size(1)
        return W_mat * ((1.0 / math.sqrt(d_in)) / W_mat.std().clamp(min=1e-8))

    # Sections 3.3, 3.4 & 3.6: Head-Coupled GQA SVD + Thin QR SwiGLU Basis + Variance Norm
    for l_idx in range(28):
        t_layer = teacher.model.layers[l_idx]
        s_layer = student.model.layers[l_idx]
        W_Q_T = t_layer.self_attn.q_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
        b_Q_T = t_layer.self_attn.q_proj.bias.detach().to(device=DEVICE, dtype=torch.float32)
        W_K_T = t_layer.self_attn.k_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
        b_K_T = t_layer.self_attn.k_proj.bias.detach().to(device=DEVICE, dtype=torch.float32)
        W_V_T = t_layer.self_attn.v_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
        b_V_T = t_layer.self_attn.v_proj.bias.detach().to(device=DEVICE, dtype=torch.float32)
        W_O_T = t_layer.self_attn.o_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)

        q_c, bq_c, k_c, bk_c, v_c, bv_c, o_c = [], [], [], [], [], [], []
        for kv_idx in range(S_num_kv_heads):
            A_K = W_K_T[kv_idx * T_head_dim : (kv_idx + 1) * T_head_dim] @ M
            U_K, _, _ = torch.linalg.svd(A_K, full_matrices=False)
            U_K_h = U_K[:, :S_head_dim]

            A_V = W_V_T[kv_idx * T_head_dim : (kv_idx + 1) * T_head_dim] @ M
            U_V, _, _ = torch.linalg.svd(A_V, full_matrices=False)
            U_V_h = U_V[:, :S_head_dim]

            k_c.append(U_K_h.t() @ A_K)
            bk_c.append(U_K_h.t() @ b_K_T[kv_idx * T_head_dim : (kv_idx + 1) * T_head_dim])
            v_c.append(U_V_h.t() @ A_V)
            bv_c.append(U_V_h.t() @ b_V_T[kv_idx * T_head_dim : (kv_idx + 1) * T_head_dim])

            for g in range(S_group_size):
                qh = kv_idx * T_group_size + g
                q_c.append(U_K_h.t() @ W_Q_T[qh * T_head_dim : (qh + 1) * T_head_dim] @ M)
                bq_c.append(U_K_h.t() @ b_Q_T[qh * T_head_dim : (qh + 1) * T_head_dim])
                o_c.append(M_pinv @ W_O_T[:, qh * T_head_dim : (qh + 1) * T_head_dim] @ U_V_h)

        W_gate_T = t_layer.mlp.gate_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
        W_up_T = t_layer.mlp.up_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
        W_down_T = t_layer.mlp.down_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
        joint_mlp = torch.cat([W_gate_T @ M, W_up_T @ M, (M_pinv @ W_down_T).t(), W_down_T.t()], dim=1)
        Q_mlp, _ = torch.linalg.qr(joint_mlp, mode="reduced")
        U_mlp = Q_mlp[:, :S_intermediate_size]

        with torch.no_grad():
            s_layer.self_attn.q_proj.weight.copy_(variance_normalize(torch.cat(q_c, 0)).to(DTYPE))
            s_layer.self_attn.k_proj.weight.copy_(variance_normalize(torch.cat(k_c, 0)).to(DTYPE))
            s_layer.self_attn.v_proj.weight.copy_(variance_normalize(torch.cat(v_c, 0)).to(DTYPE))
            s_layer.self_attn.o_proj.weight.copy_(variance_normalize(torch.cat(o_c, 1)).to(DTYPE))
            s_layer.self_attn.q_proj.bias.copy_(torch.cat(bq_c, 0).to(DTYPE))
            s_layer.self_attn.k_proj.bias.copy_(torch.cat(bk_c, 0).to(DTYPE))
            s_layer.self_attn.v_proj.bias.copy_(torch.cat(bv_c, 0).to(DTYPE))
            s_layer.mlp.gate_proj.weight.copy_(variance_normalize(U_mlp.t() @ W_gate_T @ M).to(DTYPE))
            s_layer.mlp.up_proj.weight.copy_(variance_normalize(U_mlp.t() @ W_up_T @ M).to(DTYPE))
            s_layer.mlp.down_proj.weight.copy_(variance_normalize(M_pinv @ W_down_T @ U_mlp).to(DTYPE))

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    freeze_vocab = True

elif args.init_mode == "resume":
    ckpt_path = os.path.join(args.resume_from, "resume_state.pt")
    print(f"Resuming from checkpoint: '{ckpt_path}'...")
    state = torch.load(ckpt_path, map_location=DEVICE)
    student.load_state_dict(state["model_state"])
    alpha_scalars.load_state_dict(state["alphas_state"])
    freeze_vocab = state["freeze_vocab"]

elif args.init_mode == "random":
    print("Initializing Random-Init 28L Student Baseline (100% Parameters Trainable)...")
    freeze_vocab = False

for p in student.parameters():
    p.requires_grad = True
if freeze_vocab:
    student.model.embed_tokens.weight.requires_grad = False
    student.lm_head.weight.requires_grad = False

tokenizer.padding_side = "right"
samples = []
for item in train_items:
    qp = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True)
    q_len = tokenizer([qp], return_tensors="pt").input_ids.size(1)
    ft = tokenizer.apply_chat_template(
        [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["answer"]}],
        tokenize=False,
    )
    f_ids = tokenizer([ft], return_tensors="pt", truncation=True, max_length=384).input_ids.squeeze(0)
    lbl = f_ids.clone()
    lbl[:q_len] = -100
    samples.append((f_ids, lbl))

N = len(samples)
matrix_params = [p for p in student.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    [
        {"params": matrix_params, "lr": args.lr},
        {"params": list(alpha_scalars), "lr": args.alpha_lr},
    ],
    weight_decay=0.0,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

print(f"Training on {N:,} samples (Batch={args.batch_size}, Steps={args.steps})...")
sys.stdout.flush()

for step in range(1, args.steps + 1):
    idx = torch.randperm(N)[: args.batch_size].tolist()
    chunk = [samples[i] for i in idx]
    max_l = max(x[0].size(0) for x in chunk)
    p_in = torch.full((args.batch_size, max_l), tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
    p_lbl = torch.full((args.batch_size, max_l), -100, dtype=torch.long, device=DEVICE)
    p_mask = torch.zeros((args.batch_size, max_l), dtype=torch.long, device=DEVICE)
    for i, (inp, lbl) in enumerate(chunk):
        L = inp.size(0)
        p_in[i, :L] = inp.to(DEVICE)
        p_lbl[i, :L] = lbl.to(DEVICE)
        p_mask[i, :L] = 1

    student.train()
    optimizer.zero_grad()
    out = student(input_ids=p_in, attention_mask=p_mask, labels=p_lbl)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(matrix_params + list(alpha_scalars), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if step % 100 == 0 or step == 1:
        alphas_val = torch.cat([a.detach().float() for a in alpha_scalars])
        print(
            f"   Step [{step:04d}/{args.steps}] | Loss: {out.loss.item():.4f} | "
            f"Alpha Range: [{alphas_val.min():.3f}, {alphas_val.max():.3f}]"
        )
        sys.stdout.flush()

os.makedirs(args.save_dir, exist_ok=True)
torch.save(
    {
        "model_state": student.state_dict(),
        "alphas_state": alpha_scalars.state_dict(),
        "freeze_vocab": freeze_vocab,
        "last_subject": subject_name,
    },
    os.path.join(args.save_dir, "resume_state.pt"),
)

# Fold learned scalars into linear weights for zero inference overhead
with torch.no_grad():
    for mod, alpha, handle in hook_registry:
        mod.weight.data.mul_(alpha.item())
        if mod.bias is not None:
            mod.bias.data.mul_(alpha.item())
        handle.remove()

student.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
print(f"\nSaved folded HuggingFace model (zero inference overhead) to: '{args.save_dir}'")
