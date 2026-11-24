# Zero-Student-Prior Distillation Method

[![DOI](https://zenodo.org)](https://doi.org)
[![License: CC BY 4.0](https://shields.io)](https://creativecommons.org)

Official repository for **"A Novel Distillation Method: Synthesizing All Student Parameters Mathematically from the Teacher Model with Minimal Training"**.

📄 **Preprint Paper:** [https://doi.org](https://doi.org)

---

## Overview

Conventional knowledge distillation into compact, heterogeneous student language models usually relies on an existing pre-trained small model and requires extensive gradient updates across massive vocabulary matrices ($|V| > 150{,}000$). When building a custom-width student model without any pre-trained small-model checkpoint, random initialization lacks semantic structure and fails to generalize under limited training budgets.

Our **Zero-Student-Prior Distillation Method** derives **100% of the student model's parameters mathematically from the teacher model in closed form**, followed by minimal training:

1. **Frozen Symmetric SVD Vocabulary Boundaries:** Extracts a shared orthonormal semantic trunk $\mathbf{M} \in \mathbb{R}^{d_T \times d_S}$ from the zero-centered teacher embedding matrix to synthesize and permanently freeze both input embeddings $\mathbf{E}_S$ and the language modeling head $\mathbf{W}_{\text{lm},S}$ ($50.4\%$ of total student parameters frozen, zero vocabulary training).
2. **Head-Coupled Spectral Projection for GQA:** Projects Grouped-Query Attention query, key, value, and output matrices via head-coupled SVD so per-head inner products $\mathbf{q}_h^T \mathbf{k}_h$ and residual stream directions stay aligned.
3. **Orthonormal SwiGLU Neuron Basis via Thin QR Factorization:** Builds an orthonormal SwiGLU neuron basis $\mathbf{U}_{\text{mlp}}$ via thin QR factorization of the coupled MLP matrix $\mathbf{J}_{\text{mlp}}$ to preserve dominant energy across `gate`, `up`, and `down` projections.
4. **Analytical RMSNorm Transfer:** Derives student normalization gains directly from the teacher's RMSNorm weights via squared-basis projection $\operatorname{diag}(\mathbf{M}^T \operatorname{diag}(\boldsymbol{\gamma}_T) \mathbf{M})$, with $\sqrt{d_T / d_S}$ scaling on the final readout norm.
5. **Variance Normalization with Foldable Scalars:** Applies analytical variance normalization ($1/\sqrt{d_{\text{in}}}$) alongside lightweight layer-wise learnable scalars $\alpha_{l,m}$ that merge directly into linear weights at export with **zero inference overhead**.

All projection bases are strictly column-orthonormal and achieve the theoretical optimal spectral condition number $\kappa_2 = 1.0000$.

---

## Key Results (Qwen2.5-7B-Instruct $\to$ 0.54B Student)

| Model | Layers ($L$) | Width ($d$) | MLP ($m$) | Frozen Vocab | Final Loss $\downarrow$ | Inference Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-7B Teacher | 28 | 3584 | 18944 | — | — | — |
| 24L Naive Truncation Baseline | 24 | 896 | 4864 | 100% Frozen | 0.5754 | 0 |
| 28L + Analytical RMSNorm Transfer | 28 | 896 | 4864 | 100% Frozen | 0.5210 | 0 |
| **Ours (Zero-Prior Student, Full)** | **28** | **896** | **4864** | **100% Frozen** | **0.4981** | **0 (Folded)** |

---


## Citation

If you find this work helpful in your research, please cite our preprint:

```bibtex
@misc{zou2026zerostudentprior,
  author       = {Zou, Dingguo},
  title        = {A Novel Distillation Method: Synthesizing All Student Parameters Mathematically from the Teacher Model with Minimal Training},
  year         = {2026},
  month        = nov,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24512092},
  url          = {https://doi.org}
}
```
