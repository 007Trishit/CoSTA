# CoSTA: COntractive STAbilization of Deep Reconstruction Operators

Official code for **"Stabilizing Deep Reconstruction Operators with Contractive Anchoring"** (ECCV 2026).

> 📄 **Paper:** *coming soon*  &nbsp;|&nbsp;  📚 **arXiv:** *coming soon*  &nbsp;|&nbsp;  🏷️ Arghya Sinha, Trishit Mukherjee, Kunal N. Chaudhary — Indian Institute of Science, Bangalore

---

## Overview

Pretrained Gaussian denoisers can solve a wide range of model-based image reconstruction problems through **Plug-and-Play (PnP)** and **Regularization-by-Denoising (RED)** without per-task retraining. But these denoisers are trained only for *single-step* denoising, so iterating them inside a proximal solver often produces the familiar **peak-and-collapse** failure: metrics like PSNR climb for a while, hit a peak, then abruptly degrade.

**CoSTA** is a data-driven, drop-in stabilizer that prevents this collapse **without retraining or modifying the black-box operator**. The idea:

- Treat the (possibly unstable) reconstruction operator `T` purely as a black box — only input/output evaluations are used.
- Measure local expansiveness with a **stability-index** `η`, computed with respect to the fixed point `p` of a contractive anchor.
- Blend `T` with a lightweight **contractive anchor** `S` via `T_θ = (1−θ)T + θS`, choosing `θ` adaptively so the stabilized operator stays non-expansive when needed.
- `θ` comes from a **closed-form quadratic root** of `η(x, T_θ)² = 1`, so there are no extra hyperparameters to tune.

The result is a wrapper that is agnostic to the proximal solver (PGD / HQS / ADMM / RED-GD), the pretrained denoiser (CNN / diffusion / transformer), and the measurement model.

---

## Method at a glance

```
xₖ₊₁ = T_θₖ(xₖ),     T_θ = (1−θ)·T + θ·S

ηₖ   = ‖T(xₖ) − p‖ / ‖xₖ − p‖          # stability-index at xₖ
θₖ   = minimal root of  η(xₖ, T_θ)² = 1   if ηₖ > 1   else   0
```

`p` is the fixed point of the contractive anchor `S`. When `T` behaves (`η ≤ 1`), `θ = 0` and the update is *exactly* the original black-box operator. Only when `T` starts to expand does `θ` activate, just enough to pull the iterate back toward stability while keeping the update largely driven by `T` (so reconstruction quality is preserved).

---

## Repository structure

```
CoSTA/
├── demo1.py                       # Deblurring demo (DRUNet, Levin kernel 3, σ=0.03)
├── demo2.py                       # Deblurring demo (DRUNet, Levin kernel 2, σ=0.02)
├── deepinv_denoiser.py            # Black-box denoiser wrappers (DRUNet + equivariant) via DeepInverse
├── requirements.txt
│
├── classes/
│   ├── PnP_class.py                          # Base PnP / forward-model machinery
│   ├── OursStabilizingMethod_PnP_class.py    # CoSTA stabilizer (η, θ, anchor blending)
│   ├── OursStabilizingMethod_PnP_class_v2.py # Variant of the stabilizer
│   ├── blur_utils.py                         # Blur forward-model utilities
│   ├── utils_restoration.py                  # Image I/O helpers
│   └── utils_image.py                        # Image-processing utilities
│
├── denoisers/                     # Contractive anchor (CCD) building blocks
│   ├── ccd.py                     # Contractive Convex Denoiser — the neural anchor S
│   ├── multi_conv.py              # Nonexpansive multi-convolution operator W
│   ├── linear_spline.py           # Learnable monotone linear-spline activation
│   ├── quadratic_spline.py
│   ├── spline_module.py           # Noise-level scaling spline ν(σ)
│   └── spline_autograd_func.py    # Custom autograd for the splines
│
├── pretrained/
│   ├── ccd_model.pth              # ✅ Contractive anchor weights (ships with repo)
│   └── ccd_config.yml             # Anchor architecture / training config
│
└── images/
    ├── leaves.png, im_078.png     # Sample test images
    └── kernels/                   # Blur kernels (Levin09.mat, kernels_12.mat)
```

---

## The contractive anchor (non-conventional design notes)

The anchor `S = D_σ ∘ prox_{ρf}` is contractive **by construction** because the denoiser `D_σ` is. `D_σ` (in `denoisers/ccd.py`) is a single-layer *gradient-step* denoiser

```
D_σ(x) = (1 − γτ)·x − γ·Wᵀ ( ν(σ)⁻¹ · ψ( ν(σ)·W x ) )
```

A few deliberate, non-standard choices make this both expressive and provably contractive:

- **Gradient-step parameterization.** `D` is the gradient step of a smooth, strongly convex potential `φ(x) = Σⱼ φⱼ((Wx)ⱼ) + (τ/2)‖x‖²`. Strong convexity + smoothness of `φ` guarantee `D` is a κ-contraction for a small enough step size, so we don't need spectral-normalization tricks at inference.
- **Nonexpansive convolution `W`.** `W` is reparameterized as `W = W̃·R^{−1/2}` with a row-sum normalization that guarantees `‖W‖₂ ≤ 1`. This lets the 3×3 / 64-channel filters train **freely** while preserving `‖W‖ ≤ 1` throughout training and downstream use.
- **Monotone linear-spline activations.** Each nonlinearity is a learnable linear spline with slopes constrained to `[0, 1]` (101 equidistant knots). Because each `ψⱼ = φⱼ′` with `φⱼ` convex, the activation is monotone — which is exactly what keeps the potential convex. We found this more expressive than ReLU for this role.
- **Noise-aware scaling `ν(σ)`.** A small spline-based scaling conditions the *same* anchor on a range of noise levels, so one model handles `σ ∈ [0, 25/255]` rather than training a separate anchor per noise level.
- **Contraction budgeting via `τ`.** With the fixed step size `γ = 1/(1+2τ)`, the contraction factor is `κ = √(1+2τ+2τ²)/(1+2τ)`. `τ` is constrained to `[0.0102, 0.135]` so that `κ ∈ [0.9, 0.99]` — strong enough to anchor, weak enough to stay expressive.

On the stabilizer side, the **closed-form θ** (solving `η² = 1` and taking the minimal root in `[0,1]`) is the other non-obvious choice: it makes CoSTA a true drop-in with no per-run tuning, instead of an iterative search over the blend weight.

---

## Installation

```bash
# 1. Environment
conda create -n costa python=3.10 -y
conda activate costa

# 2. PyTorch (match your CUDA version — see https://pytorch.org)
pip install torch torchvision

# 3. Remaining dependencies
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; the demos default to `cuda:0` (switch to `cpu` in the demo script if needed).

---

## Pretrained models

The **contractive anchor** weights (`pretrained/ccd_model.pth`) are included in this repository.

The **black-box denoiser** weights used in the experiments (DRUNet and others) are hosted on Google Drive:

➡️ **[Pretrained denoisers (Google Drive)](https://drive.google.com/drive/folders/1D7gRmjC36a9ZDvfDrWjJxxTP6JA1ddn7?usp=sharing)**

After downloading, place the weights under `pretrained/` so the paths in `deepinv_denoiser.py` resolve, e.g.:

```
pretrained/
├── ccd_model.pth          # included
├── ccd_config.yml         # included
└── drunet_color.pth       # download from the Drive link above
```

---

## Running the demos

```bash
python demo1.py    # deblurring: leaves.png, Levin kernel 3, σ = 0.03
python demo2.py    # deblurring: im_078.png, Levin kernel 2, σ = 0.02
```

Each demo runs three variants of PnP-HQS side by side — **Vanilla PnP**, **Equivariant PnP**, and **Ours (CoSTA-stabilized)** — for 1001 iterations, with live-updating terminal plots of:

- **PSNR** over iterations (Ours vs. Vanilla vs. Equivariant),
- the **stability-index `η`** (black-box vs. stabilized), and
- the adaptive blend weight **`θ`**.

Results (ground truth, observation, initialization, and final reconstruction) are written to `demo_logs/`.

To try your own settings, edit the call in the demo script — `forward_model_args` (kernel, scale), `noise_level`, `num_iterations`, the stabilizer's `step_size`/`algo`, and `algo_params` (solver `name`, `step_size`, etc.).

---

## Roadmap

This release focuses on the deblurring demo with PnP-HQS. We will be extending the code soon to cover:

- **Super-resolution** (the bicubic-initialized SR pipeline from the paper),
- **Additional PnP and RED algorithms** (PGD, ADMM, RED-GD, …) and more black-box denoiser backbones,
- Updated **paper and arXiv links** once available.

---

## Citation

If you find this work useful, please cite it (BibTeX to be updated once the proceedings/arXiv entry is live):

```bibtex
@inproceedings{sinha2026costa,
  title     = {Stabilizing Deep Reconstruction Operators with Contractive Anchoring},
  author    = {Sinha, Arghya and Mukherjee, Trishit and Chaudhary, Kunal N.},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## Acknowledgements

Pretrained denoisers and several utilities build on [DeepInverse](https://deepinv.github.io/deepinv). The contractive denoiser design draws on gradient-step / convex-regularizer denoisers and Lipschitz-constrained convolution parameterizations from the works cited in the paper.
