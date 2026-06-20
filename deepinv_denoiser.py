import torch
import numpy as np
import deepinv as dinv

initialized_denoisers = {}


def get_denoiser(denoiser_name, device='cuda', n_channels=3):
    if denoiser_name in initialized_denoisers:
        return initialized_denoisers[denoiser_name]

    if denoiser_name == 'DRUNet':
        path = "pretrained/drunet_color.pth" if n_channels == 3 else "pretrained/drunet_gray.pth"
        denoiser = dinv.models.DRUNet(in_channels=n_channels, out_channels=n_channels, pretrained=path, device=device)
    else:
        raise ValueError(f"Denoiser '{denoiser_name}' not recognized")

    initialized_denoisers[denoiser_name] = (run_denoiser_np, denoiser)
    return run_denoiser_np, denoiser


def run_denoiser_np(y, denoiser, sigma=0.05, device='cuda', **kwargs):
    y_ = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    x_ = denoiser(y_, sigma)
    return x_.cpu().squeeze(0).detach().numpy().astype(np.float32)


def EquivDen(y, denoiser, sigma, random=False, device='cuda'):
    y_ = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    equiv = dinv.models.EquivariantDenoiser(denoiser=denoiser, random=random)
    x_ = equiv(y_, sigma)
    return x_.cpu().squeeze(0).detach().numpy().astype(np.float32)
