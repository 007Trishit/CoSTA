import torch
import torch.nn as nn
import numpy as np
import deepinv as dinv

initialized_denoisers = {}


def get_denoiser(denoiser_name, device='cuda', n_channels=3):
    if denoiser_name in initialized_denoisers:
        return initialized_denoisers[denoiser_name]

    run_function = run_denoiser_np

    if denoiser_name == 'DRUNet':
        path = "pretrained/drunet_color.pth" if n_channels == 3 else "pretrained/drunet_gray.pth"
        denoiser = dinv.models.DRUNet(in_channels=n_channels, out_channels=n_channels,
                                      pretrained=path, device=device)

    elif denoiser_name == 'DnCNN':
        path = "pretrained/dncnn_sigma2_color.pth" if n_channels == 3 else "pretrained/dncnn_sigma2_gray.pth"
        denoiser = dinv.models.DnCNN(in_channels=n_channels, out_channels=n_channels,
                                     pretrained=path, device=device)

    elif denoiser_name == 'MMO':
        # Maximally-monotone operator (Pesquet et al.): a Lipschitz-constrained DnCNN.
        path = "pretrained/dncnn_sigma2_lipschitz_color.pth" if n_channels == 3 else "pretrained/dncnn_sigma2_lipschitz_gray.pth"
        denoiser = dinv.models.DnCNN(in_channels=n_channels, out_channels=n_channels,
                                     pretrained=path, device=device)

    elif denoiser_name == 'GSDRUNet':
        path = "pretrained/GSDRUNet_color_torch.ckpt" if n_channels == 3 else "pretrained/GSDRUNet_grayscale_torch.ckpt"
        denoiser = dinv.models.GSDRUNet(alpha=1.0, in_channels=n_channels, out_channels=n_channels,
                                        nb=2, nc=[64, 128, 256, 512], act_mode='E',
                                        pretrained=path).to(device)

    elif denoiser_name == 'DiffUNet':
        denoiser = dinv.models.DiffUNet(in_channels=3, out_channels=3, large_model=False,
                                        use_fp16=False,
                                        pretrained='pretrained/diffusion_ffhq_10m.pt').to(device)
        run_function = run_diffunet

    elif denoiser_name.startswith('CoCo'):
        # CoCoDRUNet: cocoercive/conservative denoiser on a UNetRes backbone (noise-map input).
        from denoisers.network_unet import UNetRes
        path = "pretrained/coco_color.pth" if n_channels == 3 else "pretrained/coco_gray.pth"
        denoiser = UNetRes(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4,
                           act_mode='R', downsample_mode="strideconv",
                           upsample_mode="convtranspose", bias=False)
        denoiser.load_state_dict(torch.load(path, map_location=device))
        denoiser = denoiser.to(device)
        run_function = run_unetres_denoiser_np

    elif 'PC' in denoiser_name:
        # (S)PC-DRUNet: (strictly) pseudo-contractive denoiser on a UNetRes backbone.
        from denoisers.network_unet import UNetRes
        if 'SPC' in denoiser_name:
            path = "pretrained/SPC_DRUNet_color.pth" if n_channels == 3 else "pretrained/SPC_DRUNet_gray.pth"
        else:
            path = "pretrained/PC_DRUNet_color.pth" if n_channels == 3 else "pretrained/PC_DRUNet_gray.pth"
        denoiser = UNetRes(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4,
                           act_mode='R', downsample_mode="strideconv",
                           upsample_mode="convtranspose", bias=False)
        denoiser.load_state_dict(torch.load(path, map_location=device))
        denoiser = denoiser.to(device)
        run_function = run_unetres_denoiser_np

    else:
        raise ValueError(f"Denoiser '{denoiser_name}' not recognized")

    initialized_denoisers[denoiser_name] = (run_function, denoiser)
    return run_function, denoiser


def run_denoiser_np(y, denoiser, sigma=0.05, device='cuda', **kwargs):
    y_ = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    x_ = denoiser(y_, sigma)
    return x_.cpu().squeeze(0).detach().numpy().astype(np.float32)


def run_diffunet(y, denoiser, sigma=0.05, device='cuda', **kwargs):
    y_ = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    x_ = denoiser.forward_denoise(y_, sigma)
    return x_.cpu().squeeze(0).detach().numpy().astype(np.float32)


def run_unetres_denoiser_np(y, denoiser, sigma=0.05, device='cuda', **kwargs):
    # UNetRes denoisers (CoCo / SPC / PC-DRUNet) take a concatenated noise-level map
    # and require spatial dims divisible by 8, so we pad/crop around the forward pass.
    y_ = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    h, w = y_.size(2), y_.size(3)
    pad_b = int(np.ceil(h / 8) * 8 - h)
    pad_r = int(np.ceil(w / 8) * 8 - w)
    if pad_b > 0 or pad_r > 0:
        y_ = nn.ReplicationPad2d((0, pad_r, 0, pad_b))(y_)

    noise_map = torch.ones((y_.size(0), 1, y_.size(2), y_.size(3))) * sigma
    y_ = torch.cat((y_, noise_map), dim=1).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        x = denoiser(y_)
    x = x[..., :h, :w]
    return x.cpu().squeeze(0).detach().numpy().astype(np.float32)


def EquivDen(y, denoiser, sigma, random=False, device='cuda'):
    y_ = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)
    equiv = dinv.models.EquivariantDenoiser(denoiser=denoiser, random=random)
    x_ = equiv(y_, sigma)
    return x_.cpu().squeeze(0).detach().numpy().astype(np.float32)