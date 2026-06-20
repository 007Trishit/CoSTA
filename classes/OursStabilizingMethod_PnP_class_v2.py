"""
Streamlined PnP runner: Vanilla, ViSTA (Ours), and Equivariant in a single loop.

Specialized for the (dynamic_anchor=False, dynamic_cf=True) configuration.
Implements proper CUDA-synchronized timing and saves per-iteration timings to CSV
alongside norms and PSNRs.
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classes.PnP_class import *
from denoisers.ccd import CCD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_ccd_model(model_args):
    return CCD(**model_args)


def cuda_sync(device):
    """Synchronize CUDA so that subsequent time.time() reflects completed GPU work.

    No-op on CPU. Without this, async kernel launches make wall-clock timings
    severely under-report actual GPU compute.
    """
    if isinstance(device, torch.device):
        is_cuda = device.type == 'cuda'
    else:
        is_cuda = isinstance(device, str) and device.startswith('cuda')
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class img_OursStabilizingMethod_PnP_v2(img_PnP):
    """Restricted to dynamic_anchor=False and dynamic_cf=True.

    Runs Vanilla PnP, ViSTA/Ours PnP (with stabilizer), and (optionally)
    Equivariant PnP in one shared loop. Records per-iteration wall-clock
    times for each method (CUDA-synced) and writes them to CSV.
    """

    # ----- Stabilizer (CCD) call ----------------------------------------------

    def run_ccd(self, y, denoiser, sigma=0.01):
        y_ = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(self.device)
        x_ = denoiser(y_, sigma)
        return x_.cpu().squeeze(0).detach().numpy().astype(np.float32)

    def stabilizer(self, x, stabilizer_id, stabilizer_args):
        neu = stabilizer_args.get('noise_factor', 2) * self.noise_level

        if not hasattr(self, 'ccd') or stabilizer_args['path'] != self.stabilizer_path:
            if hasattr(self, 'ccd'):
                del self.ccd
                torch.cuda.empty_cache()

            self.stabilizer_name = stabilizer_args.get('name', 'CCD')
            path = stabilizer_args['path']
            print(f"Loading {self.stabilizer_name} model from: {path}")
            cfg_path = os.path.join(
                os.path.dirname(path),
                self.stabilizer_name.lower() + '_config.yml',
            )
            with open(cfg_path, 'r') as f:
                config = yaml.safe_load(f)
            self.ccd = build_ccd_model(config['model']).to(self.device)
            if not stabilizer_args.get('rw', False):
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                self.ccd.load_state_dict(checkpoint['model_state_dict'])
            if stabilizer_args.get('ne', False):
                with torch.no_grad():
                    self.ccd.tau.data.fill_(0.0)
            self.ccd.eval()
            print("CCD model loaded from:", path)
            self.stabilizer_path = path

        step_size = stabilizer_args.get('step_size', 8)
        algo_name = stabilizer_args.get('algo', 'HQS')
        method = getattr(self, algo_name)
        with torch.no_grad():
            y = method((x, 0, 0), step_size, self.run_ccd, {'sigma': neu}, self.ccd)[0]
        return y.copy()

    # ----- Fixed point of the stabilizer -------------------------------------

    def set_fixed_point(self, stabilizer_id, stabilizer_args, steps=100):
        start_time = time.time()
        self.stablerFP = self.start_image.copy()
        for _ in range(steps):
            self.stablerFP = self.stabilizer(self.stablerFP, stabilizer_id, stabilizer_args)
        self.stablerFP = self.stablerFP.astype(np.float32)
        cuda_sync(self.device)
        print(f"Fixed point set in {time.time() - start_time:.2f} s")
        print("Fixed point PSNR:", psnr(self.image, self.stablerFP, data_range=1.0))

    # ----- Theta computation (closed-form) -----------------------------------

    def calculate_theta(self, xx, yy, ww, eta):
        """xx <- T(x) (PnP step), yy <- S(x) (stabilizer), ww <- x (previous y)."""
        pp = self.stablerFP
        A = np.linalg.norm(xx.ravel() - yy.ravel()) ** 2
        C = np.linalg.norm(xx.ravel() - pp.ravel()) ** 2 - np.linalg.norm(ww.ravel() - pp.ravel()) ** 2
        B = 2 * np.sum((yy.ravel() - xx.ravel()) * (xx.ravel() - pp.ravel()))

        if A < 1e-20:
            return 0.0

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return 0.0

        sqrt_disc = np.sqrt(discriminant)
        theta1 = (-B + sqrt_disc) / (2 * A)
        theta2 = (-B - sqrt_disc) / (2 * A)
        roots = [t for t in (theta1, theta2) if 0.0 <= t <= 1.0]
        if not roots:
            if eta > 1:
                print(f"Warning: no theta in [0,1] for eta={eta:.4f} ({theta1:.4f}, {theta2:.4f})")
            return 0.0
        return min(roots)

    # ----- Main runner --------------------------------------------------------

    def OursStabilizingMethod_PnP(self,
                                  denoiser, denoiser_args, denoiser_object=None,
                                  num_iterations=10, plot_graphs=True, plot_interval=100,
                                  equivariant=False, random=True, equiv_object=None,
                                  stabilizer_id='ccd', stabilizer_args=None,
                                  algo_params=None, **kwargs):
        """
        Runs Vanilla, ViSTA/Ours, and (optional) Equivariant PnP in a shared loop.

        Hardcoded to dynamic_anchor=False, dynamic_cf=True (the dynamic-mu, fixed-anchor
        regime). Stabilizer fixed point is computed once before the loop.
        """
        if stabilizer_args is None:
            stabilizer_args = {}
        if algo_params is None:
            algo_params = {'transpose': True, 'name': 'FBS', 'step_size': 1.9, 'clip': False}

        self.algo_params = algo_params
        if 'name' not in algo_params:
            raise ValueError("'name' missing from algo_params")
        method = getattr(self, algo_params['name'])
        denoiser_args['device'] = self.device
        step_size = algo_params.get('step_size', 1.9)

        # Merge stabilizer args with defaults
        stab_defaults = {
            'relax': 1, 'mu': 0.95, 'theta': 0.0, 'step_size': 1.9, 'algo': 'FBS',
            'noise_factor': 2, 'path': None, 'rw': False, 'ne': False, 'name': 'CCD',
            'calc_theta_roots': True, 'THETA': 0.1,
        }
        stab_defaults.update(stabilizer_args)
        stabilizer_args = stab_defaults

        print("using stabilizer:", stabilizer_id)
        self.stabilizer_path = stabilizer_args.get('path', None)

        # Pre-compute the stabilizer fixed point (dynamic_anchor=False)
        self.set_fixed_point(stabilizer_id, stabilizer_args)

        # ---- iteration state ----
        y       = self.start_image.copy();  y_old   = y.copy()
        yD      = self.start_image.copy();  yD_old  = yD.copy()
        yD2     = self.start_image.copy();  yD2_old = yD2.copy()

        state   = (y.copy(),   y.copy(),   0)
        stateD  = (yD.copy(),  yD.copy(),  0)
        stateD2 = (yD2.copy(), yD2.copy(), 0)

        N, ND, ND2 = [], [], []
        all_iters, all_itersD, all_itersD2 = [y.copy()], [yD.copy()], [yD2.copy()]

        psnr0, ssim0 = self.get_metrics(y)
        psnrs   = [psnr0]; ssims   = [ssim0]
        psnrsD  = [psnr0]; ssimsD  = [ssim0]
        psnrsD2 = [psnr0]; ssimsD2 = [ssim0]

        all_Thetas, all_unstb_etas, all_stb_etas, all_cfs = [], [], [], []

        # ---- per-iteration timing buffers ----
        # Total time per iteration (PnP step + stabilizer + theta + combine)
        ours_pnp_times    = []   # ViSTA total
        # Sub-components for Ours, useful for analysis & for parallelization speedup estimates
        ours_pnp_step_times    = []   # just the PnP T(x) step
        ours_stab_step_times   = []   # just the stabilizer S(x) step
        ours_overhead_times    = []   # theta calc + combine + bookkeeping (CPU/numpy)

        vanilla_pnp_times = []
        equiv_pnp_times   = []

        ours_total = vanilla_total = equiv_total = 0.0

        for i in range(num_iterations):

            # =========================================================
            # OURS (ViSTA): PnP step T(x), stabilizer S(x), then combine
            # =========================================================

            # --- T(x): PnP step ---
            cuda_sync(self.device)
            t0 = time.time()
            new_state = method(state, step_size, denoiser, denoiser_args, denoiser_object)
            x = new_state[0].copy()
            cuda_sync(self.device)
            t_pnp_step = time.time() - t0
            ours_pnp_step_times.append(t_pnp_step)

            # --- S(x): stabilizer step + dynamic mu (eta_unst uses CPU norms; fast) ---
            cuda_sync(self.device)
            t0 = time.time()
            eta_unst = (np.linalg.norm(x.ravel() - self.stablerFP.ravel())
                        / max(np.linalg.norm(y.ravel() - self.stablerFP.ravel()), 1e-12))
            z = self.stabilizer(state[0], stabilizer_id, stabilizer_args)
            mu = (np.linalg.norm(z.ravel() - self.stablerFP.ravel())
                  / max(np.linalg.norm(y.ravel() - self.stablerFP.ravel()), 1e-12))
            cuda_sync(self.device)
            t_stab_step = time.time() - t0
            ours_stab_step_times.append(t_stab_step)

            all_unstb_etas.append(eta_unst)
            all_cfs.append(mu)

            # --- Theta + combine (CPU/numpy) ---
            t0 = time.time()
            if i > stabilizer_args['relax']:
                Theta_k = self.calculate_theta(x, z, y, eta_unst)
            else:
                Theta_k = stabilizer_args['theta']
            y = (1.0 - Theta_k) * x + Theta_k * z
            t_overhead = time.time() - t0
            ours_overhead_times.append(t_overhead)

            ours_pnp_times.append(t_pnp_step + t_stab_step + t_overhead)
            ours_total += ours_pnp_times[-1]

            state = (y.copy(), y.copy(), y.copy())
            all_Thetas.append(Theta_k)

            eta_stb = (np.linalg.norm(y.ravel() - self.stablerFP.ravel())
                       / max(np.linalg.norm(y_old.ravel() - self.stablerFP.ravel()), 1e-12))
            all_stb_etas.append(eta_stb)

            N.append(np.linalg.norm(y.ravel() - y_old.ravel()))
            psnr_v, ssim_v = self.get_metrics(y)
            psnrs.append(psnr_v); ssims.append(ssim_v)
            y_old = y.copy()
            all_iters.append(y.copy())

            # =========================================================
            # VANILLA PnP
            # =========================================================
            cuda_sync(self.device)
            t0 = time.time()
            stateD = method(stateD, step_size, denoiser, denoiser_args, denoiser_object)
            yD = stateD[0].copy()
            cuda_sync(self.device)
            t_van = time.time() - t0
            vanilla_pnp_times.append(t_van)
            vanilla_total += t_van

            ND.append(np.linalg.norm(yD.ravel() - yD_old.ravel()))
            psnr_v, ssim_v = self.get_metrics(yD)
            psnrsD.append(psnr_v); ssimsD.append(ssim_v)
            yD_old = yD.copy()
            all_itersD.append(yD.copy())

            # =========================================================
            # EQUIVARIANT PnP (optional)
            # =========================================================
            if equivariant:
                if equiv_object is None:
                    raise ValueError("equiv_object must be provided when equivariant=True")
                equiv_args = {**denoiser_args, 'random': random}

                cuda_sync(self.device)
                t0 = time.time()
                stateD2 = method(stateD2, step_size, equiv_object, equiv_args, denoiser_object)
                yD2 = stateD2[0].copy()
                cuda_sync(self.device)
                t_eq = time.time() - t0
                equiv_pnp_times.append(t_eq)
                equiv_total += t_eq

                ND2.append(np.linalg.norm(yD2.ravel() - yD2_old.ravel()))
                psnr_v, ssim_v = self.get_metrics(yD2)
                psnrsD2.append(psnr_v); ssimsD2.append(ssim_v)
                yD2_old = yD2.copy()
                all_itersD2.append(yD2.copy())

            # ---- live diagnostics ----
            if plot_graphs and (i % plot_interval == 0) and i > 1:
                print(f"[iter {i:4d}] "
                      f"Theta={Theta_k:.4f}  eta_unst={eta_unst:.4f}  "
                      f"eta_stb={eta_stb:.4f}  mu={mu:.4f}")
                print(f"           PSNR  Ours={psnrs[-1]:.2f}  "
                      f"Vanilla={psnrsD[-1]:.2f}"
                      + (f"  Equiv={psnrsD2[-1]:.2f}" if equivariant else ""))
                print(f"           avg/iter  Ours={ours_total/(i+1):.4f}s  "
                      f"Vanilla={vanilla_total/(i+1):.4f}s"
                      + (f"  Equiv={equiv_total/(i+1):.4f}s" if equivariant else ""))

        # ---- final summary ----
        n = num_iterations
        print("\n=== Timing summary (seconds) ===")
        print(f"Vanilla PnP : total={vanilla_total:.3f}  mean/iter={np.mean(vanilla_pnp_times):.4f}")
        print(f"Ours (ViSTA): total={ours_total:.3f}  mean/iter={np.mean(ours_pnp_times):.4f}")
        print(f"   |- T(x) step      : mean={np.mean(ours_pnp_step_times):.4f}")
        print(f"   |- S(x) stabilizer: mean={np.mean(ours_stab_step_times):.4f}")
        print(f"   |- theta + combine: mean={np.mean(ours_overhead_times):.4f}")
        if equivariant:
            print(f"Equivariant : total={equiv_total:.3f}  mean/iter={np.mean(equiv_pnp_times):.4f}")

        # ---- save outputs ----
        self.reconstruction         = y.copy()
        self.reconstructionVanilla  = yD.copy()
        self.reconstructionEquiv    = yD2.copy() if equivariant else None

        # Pad sub-component lists with NaN at index 0 so they align with psnrs/norms? 
        # We keep them at length num_iterations (one per iteration) and the main 
        # psnr/norm series remain length num_iterations+1 (initial value + per-iter).
        time_dict = {
            'Vanilla': vanilla_pnp_times,
            'AVS': ours_pnp_times,
            'AVS_Tstep': ours_pnp_step_times,
            'AVS_Sstep': ours_stab_step_times,
            'AVS_overhead': ours_overhead_times,
        }
        norm_dict = {'AVS': N, 'Vanilla': ND}
        psnr_dict = {'AVS': psnrs, 'Vanilla': psnrsD}

        if equivariant:
            time_dict['Equiv'] = equiv_pnp_times
            norm_dict['Equiv'] = ND2
            psnr_dict['Equiv'] = psnrsD2

        os.makedirs(self.save_path, exist_ok=True)
        # PSNR series have length num_iterations+1; norm/time series have length num_iterations.
        # pandas handles unequal lengths via separate frames.
        pd.DataFrame(time_dict).to_csv(os.path.join(self.save_path, 'times.csv'),  index_label='iter')
        pd.DataFrame(norm_dict).to_csv(os.path.join(self.save_path, 'norms.csv'),  index_label='iter')
        pd.DataFrame(psnr_dict).to_csv(os.path.join(self.save_path, 'psnrs.csv'),  index_label='iter')
        pd.DataFrame({
            'Theta': all_Thetas,
            'eta_unst': all_unstb_etas,
            'eta_stb': all_stb_etas,
            'mu': all_cfs,
        }).to_csv(os.path.join(self.save_path, 'diagnostics.csv'), index_label='iter')

        # Quick PSNR-vs-iter PNG
        try:
            plt.figure(figsize=(6, 3))
            plt.plot(psnrs,  label='Ours (ViSTA)')
            plt.plot(psnrsD, label='Vanilla', linestyle='--')
            if equivariant:
                plt.plot(psnrsD2, label='Equivariant', linestyle='-.')
            plt.xlabel('iteration'); plt.ylabel('PSNR (dB)')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, 'psnrs.png'), dpi=120)
            plt.close()
        except Exception as e:
            print(f"Could not save PSNR plot: {e}")

        self.norm_dict = norm_dict
        self.psnr_dict = psnr_dict
        self.time_dict = time_dict

        self.get_images(save=True)
