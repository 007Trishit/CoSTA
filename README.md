# COSTA

**Stabilizing Deep Reconstruction Operators with Contractive Anchoring** — ECCV 2026
Arghya Sinha, Trishit Mukherjee, Kunal N. Chaudhury · Indian Institute of Science, Bangalore

📦 **Repo:** https://github.com/trishitmg/costa &nbsp;·&nbsp; 📄 Paper & arXiv links coming soon.
**Status:** deblurring and super-resolution demos are live; more solvers, denoisers, and reproduction scripts are being added.

---

## What COSTA does

Pretrained denoisers are trained for a single denoising step, so reusing them inside Plug-and-Play (PnP) or Regularization-by-Denoising (RED) solvers tends to **peak and then collapse**: quality climbs for a while, hits a peak, and abruptly degrades. COSTA is a drop-in fix that keeps the trajectory stable past the peak — no retraining, no architectural change, and the underlying solver/denoiser is treated as a black box. It works by gently blending the black-box update with a small **contractive anchor** only when the iteration starts to misbehave, with the blend weight computed in closed form.

---

## Getting set up

```bash
conda create -n costa python=3.11 -y
conda activate costa

# PyTorch — pick the build matching your CUDA version (https://pytorch.org)
pip install torch torchvision

pip install -r requirements.txt
```

A CUDA GPU is recommended; the demos default to `cuda:0` (switch to `cpu` in the demo script if needed).

---

## Pretrained weights

The color contractive anchor ships with the repo. The black-box denoiser weights and the grayscale anchor live on Google Drive:

➡️ **[Download pretrained models](https://drive.google.com/drive/folders/1D7gRmjC36a9ZDvfDrWjJxxTP6JA1ddn7?usp=sharing)**

Drop everything into `pretrained/`, matching the paths in `deepinv_denoiser.py`:

```
pretrained/
├── ccd_color.pth / ccd_color_config.yml     # contractive anchor (color, ships with repo)
├── ccd_gray.pth  / ccd_gray_config.yml      # contractive anchor (grayscale)
├── drunet_color.pth / drunet_gray.pth
├── dncnn_sigma2_color.pth / dncnn_sigma2_lipschitz_color.pth   # DnCNN / MMO
├── GSDRUNet_color_torch.ckpt
├── coco_color.pth / SPC_DRUNet_color.pth
└── diffusion_ffhq_10m.pt
```

---

## Run the demos

```bash
python demo1.py    # deblurring — leaves, Levin kernel 3, black-box solver FBS
python demo2.py    # deblurring — im_078, Levin kernel 2, black-box solver HQS
python demo3.py    # 4x super-resolution — im_062, black-box HQS + DRS anchor
```

Each demo runs **Vanilla PnP**, **Equivariant PnP**, and **COSTA** side by side, with live terminal plots of PSNR, the stability-index $\eta$, and the blend weight $\theta$. Results are written to `demo_logs/`.

To adapt a demo, edit the call:

- **Task** — `forward_model_name='deblurring'` or `'superresolution'`, with `forward_model_args={'scale_factor': 4, 'kernel_id': 2, 'device': device}`.
- **Black-box denoiser** — change `get_denoiser('DRUNet', ...)` to `'DnCNN'`, `'MMO'`, `'GSDRUNet'`, `'DiffUNet'`, `'CoCoDRUNet'`, or `'SPCDRUNet'`.
- **Solver** — set `algo_params['name']` to `'FBS'`, `'HQS'`, `'RED'`, or `'DRS'`.
- **Anchor** — point `stabilizer_args['path']` to `pretrained/ccd_color.pth` or `pretrained/ccd_gray.pth` (the matching `*_config.yml` is loaded automatically).

---

## How it works

Classical image reconstruction works to retrieve $\bar{x}$ from measurements $b = A\bar{x} + n$ by solving

$$
\min_{x} f(x) + g(x), \qquad f(x) = \frac{1}{2}\lVert Ax - b \rVert^2 .
$$
 
Proximal gradient descent solves this using 

$$
x_{k+1} = \mathrm{prox}_{\gamma g}(x_k - γ∇f(x_k))
$$


PnP/RED replace the proximal/gradient step of the regularizer $g$ with a pretrained denoiser, which turns each solver (PnP-PGD/FBS, PnP-HQS, RED-GD, DRS) into a single fixed-point map $T$. The instability we target is that iterating $x_{k+1} = T(x_k)$ is not guaranteed to be non-expansive, so it can diverge after an early peak.

**Stability-index.** For a reference point $p$, we track the stability-index $\eta_p(x, T)$ — the size of a single step of $T$ relative to the current distance from $p$:

$$
\eta_p(x, T) =
\begin{cases}
\dfrac{\lVert T(x) - p \rVert}{\lVert x - p \rVert}, & x \neq p, \\
0, & x = p .
\end{cases}
$$

A large stability-index, $\eta_p > 1$, marks the iterate as locally unstable with respect to $p$ and signals a risk of collapse; $\eta_p \leqslant 1$ is the stable regime we want to maintain.

**Contractive anchoring.** Take a $\kappa$-contraction $S$ (with $\kappa < 1$) whose unique fixed point is $p$, and average it with the black box,

$$
T_\theta := (1-\theta)\,T + \theta\,S, \qquad \theta \in [0,1].
$$

A larger $\theta$ lowers the stability-index but pulls the result toward the anchor's (lower-quality) fixed point, so we use the **smallest** $\theta$ that brings the stability-index back to the stable threshold $\eta_p(x, T_\theta) = 1$. Writing $T_\theta(x) - p = (1-\theta)\,(T(x)-p) + \theta\,(S(x)-p)$, this condition is a scalar quadratic in $\theta$ with a closed-form minimal root in $[0,1]$ — no tuning.


**Algorithm 1 — COSTA**

> **Require:** black-box operator $T$, contractive anchor $S$ with fixed point $p$, initialization $x_0$.
>
> **for** $k = 0, 1, 2, \dots$ **do**
> 1. compute the stability-index $\eta_k \leftarrow \eta_p(x_k, T)$
> 2. **if** $\eta_k > 1$: set $\theta_k \leftarrow \tilde{\theta}(x_k, 1)$, the smallest root in $[0,1]$ of $\eta_p(x_k, T_\theta)^2 = 1$
> 3. **else**: set $\theta_k \leftarrow 0$
> 4. update $x_{k+1} \leftarrow T_{\theta_k}(x_k) = (1-\theta_k)T(x_k) + \theta_kS(x_k)$
>
> **end for**

When $T$ is well-behaved ($\eta_k \leqslant 1$) the update is exactly the original operator. Enforcing the target level $1$ keeps the iterates bounded, and in practice $\theta_k$ stays small, so reconstruction quality near the peak is preserved.

---

## What's included

**Applications:** deblurring (Levin09 motion kernels `0–7`, 25×25 Gaussian `8`, 9×9 box `9`) and super-resolution (`kernels_12` blur + `scale_factor` ×2/×3/×4).

**Solvers:** `FBS` (proximal gradient), `HQS`, `RED` (RED-GD), and `DRS` (Douglas–Rachford splitting — not ADMM).

**Black-box denoisers:** `DRUNet`, `DnCNN`, `MMO`, `GSDRUNet`, `DiffUNet`, `CoCoDRUNet`, `SPCDRUNet`/`PCDRUNet`. The first five load through DeepInverse; `CoCo`/`PC` use a `UNetRes` backbone with a noise-level map. Any of them can be wrapped equivariantly via `EquivDen`.

**Contractive anchor:** the CCD denoiser in color (`ccd_color.pth`) and grayscale (`ccd_gray.pth`) variants.

---

## What's in the repo

```
COSTA/
├── demo1.py / demo2.py            # deblurring demos
├── demo3.py                       # super-resolution demo
├── deepinv_denoiser.py            # get_denoiser() factory + run wrappers
├── requirements.txt
│
├── classes/
│   ├── COSTA_PnP_class.py         # forward models, FBS/HQS/RED/DRS, and the COSTA driver
│   ├── blur_utils.py              # blur / SR forward model + FFT prox
│   ├── utils_restoration.py
│   └── utils_image.py
│
├── denoisers/
│   ├── ccd.py                     # Contractive Convex Denoiser — the anchor S
│   ├── network_unet.py            # UNetRes backbone (CoCo / SPC / PC-DRUNet)
│   └── ...                        # spline / multi-conv building blocks
│
├── pretrained/
│   ├── ccd_color.pth / ccd_color_config.yml
│   └── ccd_gray.pth  / ccd_gray_config.yml
│
└── images/
    ├── CBSD10/                    # test set
    ├── leaves.png, im_078.png, im_062.png
    └── kernels/                   # Levin09.mat (deblur), kernels_12.mat (SR)
```

---

## Citing this work

```bibtex
@inproceedings{sinha2026costa,
  title     = {Stabilizing Deep Reconstruction Operators with Contractive Anchoring},
  author    = {Sinha, Arghya and Mukherjee, Trishit and Chaudhury, Kunal N.},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## Acknowledgements

This work was supported by the Government of India through the PMRF and ANRF, Qualcomm Technologies, Inc. through the Qualcomm Innovation Fellowship India, and the Kotak IISc AI-ML Centre through GPU resources.
