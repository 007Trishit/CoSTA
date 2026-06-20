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

The result is a wrapper that is agnostic to the proximal solver, the pretrained denoiser, and the measurement model.

---

## Method at a glance

```
xₖ₊₁ = T_θₖ(xₖ),     T_θ = (1−θ)·T + θ·S

ηₖ   = ‖T(xₖ) − p‖ / ‖xₖ − p‖          # stability-index at xₖ
θₖ   = minimal root of  η(xₖ, T_θ)² = 1   if ηₖ > 1   else   0
```

`p` is the fixed point of the contractive anchor `S`. When `T` behaves (`η ≤ 1`), `θ = 0` and the update is *exactly* the original black-box operator. Only when `T` starts to expand does `θ` activate, just enough to pull the iterate back toward stability while keeping the update largely driven by `T` (so reconstruction quality is preserved).

---

## What's supported

**Applications (forward models)**

| Task | Notes |
|---|---|
| **Deblurring** | Levin09 motion kernels (`kernel_id` 0–7), 25×25 Gaussian (`8`), 9×9 box (`9`) |
| **Super-resolution** | `kernels_12` blur kernels + `scale_factor` (×2 / ×3 / ×4) downsampling |

**Solvers**

Proximal Gradient / Forward-Backward Splitting (`FBS`), Half-Quadratic Splitting (`HQS`), **RED-GD** (`RED`), and **Douglas–Rachford Splitting** (`DRS`). Note that `DRS` is *not* ADMM — it is the Douglas–Rachford fixed-point form used here.

**Black-box denoisers**

All PnP denoisers from the paper experiments are wired into `deepinv_denoiser.py` via a single `get_denoiser(name, n_channels, device)` factory:

`DRUNet`, `DnCNN`, `MMO`, `GSDRUNet`, `DiffUNet`, `CoCoDRUNet`, and `SPCDRUNet` / `PCDRUNet`.

`DRUNet`/`DnCNN`/`MMO`/`GSDRUNet`/`DiffUNet` load directly through DeepInverse; `CoCo` and `PC` use a `UNetRes` backbone with a concatenated noise-level map. Any denoiser can be wrapped with an equivariant variant via `EquivDen`.

**Contractive anchor (`S`)**

The CCD anchor ships in two variants — **color** and **grayscale** — selected by config:

| Variant | Config | Weights |
|---|---|---|
| CCD color (3-channel) | `ccd_color_config.yml` | `pretrained/ccd_color.pth` |
| CCD grayscale (1-channel) | `ccd_gray_config.yml` | `pretrained/ccd_gray.pth` |

---

## Repository structure

```
CoSTA/
├── demo1.py                       # Deblurring demo (DRUNet, Levin kernel 3)
├── demo2.py                       # Deblurring demo (DRUNet, Levin kernel 2)
├── deepinv_denoiser.py            # get_denoiser() factory for all black-box denoisers
├── requirements.txt
│
├── classes/
│   ├── PnP_class.py                          # Forward models (deblurring, super-resolution)
│   │                                         #   + solvers: FBS / HQS / RED / DRS
│   ├── OursStabilizingMethod_PnP_class.py    # CoSTA stabilizer (η, θ, anchor blending)
│   ├── blur_utils.py                         # Blur / SR forward-model + prox utilities
│   ├── utils_restoration.py
│   └── utils_image.py
│
├── denoisers/                     # Denoiser architectures
│   ├── ccd.py                     # Contractive Convex Denoiser — the neural anchor S
│   ├── network_unet.py            # UNetRes backbone (CoCo / SPC / PC-DRUNet)
│   ├── deal.py, wc_conv_net.py    # DEAL, WCRR
│   ├── multi_conv.py              # Nonexpansive multi-convolution operator W
│   ├── linear_spline.py, quadratic_spline.py, spline_module.py, spline_autograd_func.py
│   └── ...                        # supporting modules
│
├── pretrained/
│   ├── ccd_color.pth              # ✅ contractive anchor, color (ships with repo)
│   ├── ccd_color_config.yml
│   ├── ccd_gray.pth               #  contractive anchor, grayscale (see Drive)
│   └── ccd_gray_config.yml
│
└── images/
    ├── leaves.png, im_078.png     # sample test images
    └── kernels/                   # Levin09.mat (deblur), kernels_12.mat (SR)
```

---

## The contractive anchor (non-conventional design notes)

The anchor `S = D_σ ∘ prox_{ρf}` is contractive **by construction** because the denoiser `D_σ` is. `D_σ` (in `denoisers/ccd.py`) is a single-layer *gradient-step* denoiser

```
D_σ(x) = (1 − γτ)·x − γ·Wᵀ ( ν(σ)⁻¹ · ψ( ν(σ)·W x ) )
```

A few deliberate, non-standard choices make this both expressive and provably contractive:

- **Gradient-step parameterization.** `D` is the gradient step of a smooth, strongly convex potential `φ(x) = Σⱼ φⱼ((Wx)ⱼ) + (τ/2)‖x‖²`. Strong convexity + smoothness guarantee `D` is a κ-contraction for a small enough step size — no spectral-normalization tricks needed at inference.
- **Nonexpansive convolution `W`.** `W` is reparameterized as `W = W̃·R^{−1/2}` with a row-sum normalization that guarantees `‖W‖₂ ≤ 1`, so the 3×3 / 64-channel filters can train **freely** while staying non-expansive.
- **Monotone linear-spline activations.** Each nonlinearity is a learnable linear spline with slopes constrained to `[0, 1]` (101 knots). Since each `ψⱼ = φⱼ′` with `φⱼ` convex, the activation is monotone — exactly what keeps the potential convex. More expressive than ReLU here.
- **Noise-aware scaling `ν(σ)`.** A small spline-based scaling conditions the *same* anchor on a range of noise levels (`σ ∈ [0, 25/255]`), so one model handles all levels.
- **Contraction budgeting via `τ`.** With `γ = 1/(1+2τ)`, the contraction factor is `κ = √(1+2τ+2τ²)/(1+2τ)`. `τ` is constrained to `[0.0102, 0.135]` so `κ ∈ [0.9, 0.99]` — strong enough to anchor, weak enough to stay expressive. These bounds are identical for the color and grayscale configs; only the channel count differs.

On the stabilizer side, the **closed-form θ** (solving `η² = 1`, taking the minimal root in `[0,1]`) is the other non-obvious choice: it makes CoSTA a true drop-in with no per-run tuning, instead of an iterative search over the blend weight.

---

## Installation

```bash
conda create -n costa python=3.11 -y
conda activate costa

# PyTorch — match your CUDA version (see https://pytorch.org)
pip install torch torchvision

pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; the demos default to `cuda:0` (switch to `cpu` in the demo script if needed).

---

## Pretrained models

The **contractive anchor** weights are small and the color variant ships with the repo. All **black-box denoiser** weights from the experiments (DRUNet, DnCNN, DiffUNet, GSDRUNet, CoCoDRUNet, MMO, SPC/PC-DRUNet, DEAL, …) and the **grayscale anchor** are hosted on Google Drive:

➡️ **[Pretrained models (Google Drive)](https://drive.google.com/drive/folders/1D7gRmjC36a9ZDvfDrWjJxxTP6JA1ddn7?usp=sharing)**

Download and place everything under `pretrained/`, matching the paths in `deepinv_denoiser.py`, e.g.:

```
pretrained/
├── ccd_color.pth / ccd_color_config.yml      # color anchor (color config ships with repo)
├── ccd_gray.pth  / ccd_gray_config.yml       # grayscale anchor
├── drunet_color.pth / drunet_gray.pth
├── dncnn_sigma2_color.pth / ...
├── GSDRUNet_color_torch.ckpt / ...
├── coco_color.pth / SPC_DRUNet_color.pth / ...
├── deal_color.pth / ...
└── ...
```

---

## Running the demos

```bash
python demo1.py    # deblurring: leaves.png, Levin kernel 3
python demo2.py    # deblurring: im_078.png, Levin kernel 2
```

Each demo runs three variants of a PnP solver side by side — **Vanilla PnP**, **Equivariant PnP**, and **Ours (CoSTA-stabilized)** — with live terminal plots of **PSNR**, the **stability-index `η`** (black-box vs. stabilized), and the adaptive blend weight **`θ`**. Outputs are written to `demo_logs/`.

To change task/solver/denoiser, edit the call in the demo script:

- **Switch to super-resolution** — set `forward_model_name='superresolution'` and pass `forward_model_args={'scale_factor': 2, 'kernel_id': 3, 'device': device}`.
- **Switch the black-box denoiser** — change `get_denoiser('DRUNet', ...)` to `'DnCNN'`, `'MMO'`, `'GSDRUNet'`, `'DiffUNet'`, `'CoCoDRUNet'`, or `'SPCDRUNet'`.
- **Switch the solver** — set `algo_params['name']` to `'FBS'`, `'HQS'`, `'RED'`, or `'DRS'`.
- **Switch the anchor** — point `stabilizer_args` to the color or grayscale CCD config/weights.

---

## Roadmap

- **Super-resolution** demos and configs (forward model already supported in `PnP_class.py`).
- Broader solver / denoiser sweeps reproducing the paper tables.
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