# Mathematical-Distillation
Official repository for "A Novel Distillation Method: Synthesizing All Student Parameters Mathematically from the Teacher Model with Minimal Training."
When compressing large language models into custom widths tailored for specific hardware memory limits, two major bottlenecks arise: a pre-trained student checkpoint of that exact size often does not exist, and training massive vocabulary matrices (vocabulary size > 150,000) from random initialization requires heavy compute and easily causes semantic drift.
In this work, I explore a Zero-Student-Prior Distillation approach that derives 100% of a compact student model's parameters mathematically from the teacher model in closed form, followed by minimal training:
🔹 Frozen Symmetric SVD Vocabulary Boundaries: Synthesizes and permanently freezes both input embeddings and the LM head (50.4% of total student parameters) using a shared orthonormal semantic trunk from the zero-centered teacher embeddings—requiring zero vocabulary training.
🔹 Head-Coupled GQA & Thin QR SwiGLU Projections: Preserves attention head inner products and dominant MLP neuron energy using strictly orthonormal bases.
🔹 Analytical RMSNorm & Foldable Scalars: Stabilizes cross-layer signal variance during minimal training and folds directly into linear weights at export with zero inference overhead.
Tested on compressing Qwen2.5-7B-Instruct into a depth-matched (d_S = 896) student under tight compute budgets, our teacher-synthesized student achieves faster convergence and preserves real semantic comprehension on unseen ARC-Easy science questions without using a single pre-trained small-model weight.
📄 Read the full paper (DOI): https://doi.org/10.5281/zenodo.24512092
#AIResearch #LargeLanguageModels #KnowledgeDistillation #ModelCompression #MachineLearning #DeepLearning #LLMs

## Citation

If you find this work helpful in your research, please cite our preprint:

```bibtex
@misc{zou2026zerostudentprior,
  author       = {Zou, James},
  title        = {A Novel Distillation Method: Synthesizing All Student Parameters Mathematically from the Teacher Model with Minimal Training},
  year         = {2026},
  month        = nov,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24512092},
  url          = {https://doi.org}
}
```
