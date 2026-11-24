import os
import json
import sys
import math
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 0. Global Setup & Configuration (Exact Match to test_variance_norm_and_alpha.py)
# ==============================================================================
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_FILE = "teacher_qa_1000.json"
SAVE_PATH = "./qwen2.5-0.5B-28L-qr-mlp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

NUM_QA = 100
BATCH_SIZE = 10
MAX_STEPS = 450
LEARNING_RATE = 3e-4
ALPHA_LR = 1e-2

if not os.path.exists(CACHE_FILE):
    raise FileNotFoundError(f"❌ '{CACHE_FILE}' not found!")

print("\n" + "="*80)
print("📏 GOLDEN RULER ABLATION #2: 28L + QR ORTHONORMAL MLP BASIS (U_mlp) 📏")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

teacher = AutoModelForCausalLM.from_pretrained(TEACHER_ID, dtype=DTYPE, device_map="auto")

# Single-Variable Change: Upgrade Student from 24 layers to 28 layers (1:1 with Teacher)
student_cfg = AutoConfig.from_pretrained(STUDENT_ID)
student_cfg.num_hidden_layers = 28
student_cfg.layer_types = ["full_attention"] * 28
student_cfg.vocab_size = teacher.config.vocab_size
student = AutoModelForCausalLM.from_config(student_cfg).to(device=DEVICE, dtype=DTYPE)

W_T_raw = teacher.model.embed_tokens.weight.detach().to(device=DEVICE, dtype=torch.float32)
vocab_size, teacher_dim = W_T_raw.shape
student_dim = 896

T_num_heads = teacher.config.num_attention_heads      # 28
T_num_kv_heads = teacher.config.num_key_value_heads   # 4
T_head_dim = teacher_dim // T_num_heads              # 128
T_group_size = T_num_heads // T_num_kv_heads          # 7

S_num_heads = student.config.num_attention_heads      # 14
S_num_kv_heads = student.config.num_key_value_heads   # 2
S_head_dim = student_dim // S_num_heads              # 64
S_group_size = S_num_heads // S_num_kv_heads          # 7

T_intermediate_size = teacher.config.intermediate_size  # 18944
S_intermediate_size = student.config.intermediate_size  # 4864

# ==============================================================================
# 1. Closed-Form Vocabulary Space (Identical to Golden Ruler)
# ==============================================================================
print("⏳ [1/4] Synthesizing Closed-Form Full-Vocabulary Space...")
mu_T = torch.mean(W_T_raw, dim=0, keepdim=True)
W_T_centered = W_T_raw - mu_T
_, _, V_t = torch.linalg.svd(W_T_centered, full_matrices=False)

M = V_t[:student_dim, :].t()
M_pinv = torch.linalg.pinv(M)

W_S_synthesized = torch.matmul(W_T_centered, M)
target_std = W_T_raw.std() * 0.15
calibrated_W_S = W_S_synthesized * (target_std / W_S_synthesized.std())

M_sq_t = M.pow(2).t()  # [896, 3584] energy projection operator
def project_rmsnorm(gamma_T_tensor):
    g_T = gamma_T_tensor.detach().to(device=DEVICE, dtype=torch.float32)
    g_S = M_sq_t @ g_T
    # Normalize mean gain to match student isotropic scale while preserving relative channel profile
    return g_S

with torch.no_grad():
    student.model.norm.weight.copy_(2.0 * project_rmsnorm(teacher.model.norm.weight).to(DTYPE))
    for l_idx in range(28):
        student.model.layers.__getitem__(l_idx).input_layernorm.weight.copy_(
            project_rmsnorm(teacher.model.layers.__getitem__(l_idx).input_layernorm.weight).to(DTYPE)
        )
        student.model.layers.__getitem__(l_idx).post_attention_layernorm.weight.copy_(
            project_rmsnorm(teacher.model.layers.__getitem__(l_idx).post_attention_layernorm.weight).to(DTYPE)
        )
    student.model.embed_tokens.weight.copy_(calibrated_W_S.to(DTYPE))
    student.lm_head.weight.copy_(calibrated_W_S.to(DTYPE))

# ==============================================================================
# 2. Per-Head Coupled SVD + Method A (1/sqrt(d_in)) Across All 28 Layers (1:1)
# ==============================================================================
print("⏳ [2/4] Initializing All 28 Layers (1:1 Teacher Mapping) with Method A Variance Norm...")

def variance_normalize(W_mat):
    d_in = W_mat.size(1)
    ideal_std = 1.0 / math.sqrt(d_in)
    current_std = W_mat.std().clamp(min=1e-8)
    return W_mat * (ideal_std / current_std)

# Single-Variable Swap: QR Orthonormal MLP Basis U_mlp [18944, 4864]

for l_idx in range(28):
    t_layer = teacher.model.layers.__getitem__(l_idx)
    s_layer = student.model.layers.__getitem__(l_idx)

    W_Q_T = t_layer.self_attn.q_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
    b_Q_T = t_layer.self_attn.q_proj.bias.detach().to(device=DEVICE, dtype=torch.float32)
    W_K_T = t_layer.self_attn.k_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
    b_K_T = t_layer.self_attn.k_proj.bias.detach().to(device=DEVICE, dtype=torch.float32)
    W_V_T = t_layer.self_attn.v_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
    b_V_T = t_layer.self_attn.v_proj.bias.detach().to(device=DEVICE, dtype=torch.float32)
    W_O_T = t_layer.self_attn.o_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)

    W_Q_S_chunks, b_Q_S_chunks = [], []
    W_K_S_chunks, b_K_S_chunks = [], []
    W_V_S_chunks, b_V_S_chunks = [], []
    W_O_S_chunks = []

    for kv_idx in range(S_num_kv_heads):
        k_slice = W_K_T[kv_idx*T_head_dim : (kv_idx+1)*T_head_dim, :]
        bk_slice = b_K_T[kv_idx*T_head_dim : (kv_idx+1)*T_head_dim]
        A_K = k_slice @ M
        U_K, _, _ = torch.linalg.svd(A_K, full_matrices=False)
        U_K_head = U_K[:, :S_head_dim]

        v_slice = W_V_T[kv_idx*T_head_dim : (kv_idx+1)*T_head_dim, :]
        bv_slice = b_V_T[kv_idx*T_head_dim : (kv_idx+1)*T_head_dim]
        A_V = v_slice @ M
        U_V, _, _ = torch.linalg.svd(A_V, full_matrices=False)
        U_V_head = U_V[:, :S_head_dim]

        W_K_S_chunks.append(U_K_head.t() @ A_K)
        b_K_S_chunks.append(U_K_head.t() @ bk_slice)
        W_V_S_chunks.append(U_V_head.t() @ A_V)
        b_V_S_chunks.append(U_V_head.t() @ bv_slice)

        for g in range(S_group_size):
            q_head_idx = kv_idx * T_group_size + g
            q_slice = W_Q_T[q_head_idx*T_head_dim : (q_head_idx+1)*T_head_dim, :]
            bq_slice = b_Q_T[q_head_idx*T_head_dim : (q_head_idx+1)*T_head_dim]
            W_Q_S_chunks.append(U_K_head.t() @ q_slice @ M)
            b_Q_S_chunks.append(U_K_head.t() @ bq_slice)

            o_slice = W_O_T[:, q_head_idx*T_head_dim : (q_head_idx+1)*T_head_dim]
            W_O_S_chunks.append(M_pinv @ o_slice @ U_V_head)

    W_Q_S = variance_normalize(torch.cat(W_Q_S_chunks, dim=0))
    b_Q_S = torch.cat(b_Q_S_chunks, dim=0)
    W_K_S = variance_normalize(torch.cat(W_K_S_chunks, dim=0))
    b_K_S = torch.cat(b_K_S_chunks, dim=0)
    W_V_S = variance_normalize(torch.cat(W_V_S_chunks, dim=0))
    b_V_S = torch.cat(b_V_S_chunks, dim=0)
    W_O_S = variance_normalize(torch.cat(W_O_S_chunks, dim=1))

    W_gate_T = t_layer.mlp.gate_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
    W_up_T = t_layer.mlp.up_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)
    W_down_T = t_layer.mlp.down_proj.weight.detach().to(device=DEVICE, dtype=torch.float32)

    joint_mlp = torch.cat([W_gate_T @ M, W_up_T @ M, (M_pinv @ W_down_T).t(), W_down_T.t()], dim=1)
    Q_mlp, _ = torch.linalg.qr(joint_mlp, mode="reduced")
    U_mlp = Q_mlp[:, :S_intermediate_size]  # [18944, 4864] exact orthonormal neuron basis

    W_gate_S = variance_normalize(U_mlp.t() @ W_gate_T @ M)
    W_up_S = variance_normalize(U_mlp.t() @ W_up_T @ M)
    W_down_S = variance_normalize(M_pinv @ W_down_T @ U_mlp)

    with torch.no_grad():
        s_layer.self_attn.q_proj.weight.copy_(W_Q_S.to(DTYPE))
        s_layer.self_attn.k_proj.weight.copy_(W_K_S.to(DTYPE))
        s_layer.self_attn.v_proj.weight.copy_(W_V_S.to(DTYPE))
        s_layer.self_attn.o_proj.weight.copy_(W_O_S.to(DTYPE))
        s_layer.self_attn.q_proj.bias.copy_(b_Q_S.to(DTYPE))
        s_layer.self_attn.k_proj.bias.copy_(b_K_S.to(DTYPE))
        s_layer.self_attn.v_proj.bias.copy_(b_V_S.to(DTYPE))
        s_layer.mlp.gate_proj.weight.copy_(W_gate_S.to(DTYPE))
        s_layer.mlp.up_proj.weight.copy_(W_up_S.to(DTYPE))
        s_layer.mlp.down_proj.weight.copy_(W_down_S.to(DTYPE))

del teacher
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==============================================================================
# 3. Attach Method C Learnable Elevation Scalars (alpha_l) for All 28 Layers
# ==============================================================================
print("⏳ [3/4] Attaching 196 Learnable Elevation Scalars (alpha_l) Across 28 Layers...")
alpha_scalars = nn.ParameterList()
hook_registry = []

for l_idx in range(28):
    s_layer = student.model.layers.__getitem__(l_idx)
    target_modules = (
        s_layer.self_attn.q_proj,
        s_layer.self_attn.k_proj,
        s_layer.self_attn.v_proj,
        s_layer.self_attn.o_proj,
        s_layer.mlp.gate_proj,
        s_layer.mlp.up_proj,
        s_layer.mlp.down_proj,
    )
    for mod in target_modules:
        alpha = nn.Parameter(torch.ones(1, device=DEVICE, dtype=DTYPE))
        alpha_scalars.append(alpha)
        def make_hook(a_param):
            def hook(module, inp, out):
                return out * a_param
            return hook
        handle = mod.register_forward_hook(make_hook(alpha))
        hook_registry.append((mod, alpha, handle))

# ==============================================================================
# 4. Train on Exact Same 100 Q&A Pairs & Print Identical Golden-Ruler Metrics
# ==============================================================================
with open(CACHE_FILE, "r", encoding="utf-8") as f:
    teacher_qa = json.load(f)[:NUM_QA]

for p in student.parameters():
    p.requires_grad = True
student.model.embed_tokens.weight.requires_grad = False
student.lm_head.weight.requires_grad = False

tokenizer.padding_side = "right"
samples = []
for q_str, a_str in teacher_qa:
    qp = tokenizer.apply_chat_template([{"role": "user", "content": q_str}], tokenize=False, add_generation_prompt=True)
    q_len = tokenizer([qp], return_tensors="pt").input_ids.size(1)
    ft = tokenizer.apply_chat_template([{"role": "user", "content": q_str}, {"role": "assistant", "content": a_str}], tokenize=False)
    f_ids = tokenizer([ft], return_tensors="pt").input_ids.squeeze(0)
    lbl = f_ids.clone()
    lbl[:q_len] = -100
    samples.append((f_ids, lbl))

N = len(samples)
matrix_params = [p for p in student.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW([
    {"params": matrix_params, "lr": LEARNING_RATE},
    {"params": list(alpha_scalars), "lr": ALPHA_LR}
], weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS, eta_min=1e-6)

print(f"\n🚀 [4/4] Training 28L Student on {N} Q&A Pairs (Batch={BATCH_SIZE}, Steps={MAX_STEPS})...")
sys.stdout.flush()

for step in range(1, MAX_STEPS + 1):
    idx = torch.randperm(N)[:BATCH_SIZE].tolist()
    chunk = [samples.__getitem__(i) for i in idx]
    max_l = max(x.__getitem__(0).size(0) for x in chunk)
    p_in = torch.full((BATCH_SIZE, max_l), tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
    p_lbl = torch.full((BATCH_SIZE, max_l), -100, dtype=torch.long, device=DEVICE)
    p_mask = torch.zeros((BATCH_SIZE, max_l), dtype=torch.long, device=DEVICE)
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

    if step % 50 == 0 or step == 1:
        alphas_val = torch.cat([a.detach().float() for a in alpha_scalars])
        print(f"   ↳ Step [{step:03d}/{MAX_STEPS}] | Loss: {out.loss.item():.4f} | Alpha Range: [{alphas_val.min():.3f}, {alphas_val.max():.3f}]")
        sys.stdout.flush()

print("\n" + "="*80)
print(f"📊 GOLDEN RULER #1 COMPLETE (28-LAYER 1:1 STUDENT) | TERMINAL LOSS: {out.loss.item():.4f}")
print("="*80)
student.eval()
for check_idx in range(2):
    q_str, _ = teacher_qa.__getitem__(check_idx)
    q_prompt = tokenizer.apply_chat_template([{"role": "user", "content": q_str}], tokenize=False, add_generation_prompt=True)
    q_ids = tokenizer([q_prompt], return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        s_gen = student.generate(q_ids, max_new_tokens=40, do_sample=False)
        s_ans = tokenizer.decode(s_gen.squeeze(0)[q_ids.size(-1):], skip_special_tokens=True)
        print(f"   🔍 Q{check_idx+1}: '{q_str}'\n      --> AI: '{s_ans.strip()}'")

with torch.no_grad():
    for mod, alpha, handle in hook_registry:
        mod.weight.data.mul_(alpha.item())
        if mod.bias is not None:
            mod.bias.data.mul_(alpha.item())
        handle.remove()

student.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n💾 Clean 28L folded model saved to: {SAVE_PATH}")








