import numpy as np
from classes.PnP_class import *
import time
import yaml
import os
import plotext as plt_term
from denoisers.ccd import CCD


def build_ccd_model(model_args):
    return CCD(**model_args)


class img_COSTA_PnP(img_PnP):

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
            with open(os.path.join(os.path.dirname(path), self.stabilizer_name.lower() + f'_{"color" if self.color_mode == "RGB" else "gray"}' + '_config.yml'), 'r') as f:
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

    def set_fixed_point(self, stabilizer_id, stabilizer_args):
        start_time = time.time()
        self.stablerFP = self.start_image.copy()
        for _ in range(100):
            self.stablerFP = self.stabilizer(self.stablerFP, stabilizer_id, stabilizer_args)
        self.stablerFP = self.stablerFP.astype(np.float32)
        print(f"Fixed point set in {time.time()-start_time:.2f} s")
        print("Fixed point PSNR:", psnr(self.image, self.stablerFP, data_range=1.0))

    def calculate_theta(self, xx, yy, ww, eta):
        pp = self.stablerFP
        A = np.linalg.norm(xx.ravel() - yy.ravel()) ** 2
        C = np.linalg.norm(xx.ravel() - pp.ravel()) ** 2 - np.linalg.norm(ww.ravel() - pp.ravel()) ** 2
        B = 2 * np.sum((yy.ravel() - xx.ravel()) * (xx.ravel() - pp.ravel()))

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return 0.0

        sqrt_disc = np.sqrt(discriminant)
        theta1 = (-B + sqrt_disc) / (2 * A)
        theta2 = (-B - sqrt_disc) / (2 * A)
        roots = [t for t in [theta1, theta2] if 0 <= t <= 1]
        if not roots:
            if eta > 1:
                print(f"Warning: No valid theta in [0,1] for eta={eta:.4f}. ({theta1:.4f}, {theta2:.4f})")
            return 0.0
        return min(roots)

    def COSTA_PnP(self, denoiser, denoiser_args, denoiser_object=None,
                                  num_iterations=10, plot_graphs=True, plot_interval=100,
                                  equivariant=False, random=True, equiv_object=None,
                                  dynamic_anchor=False, dynamic_cf=True,
                                  stabilizer_id='simple', stabilizer_args={},
                                  algo_params={'transpose': True, 'name': 'FBS', 'step_size': 1.9, 'clip': False},
                                  **kwargs):
        self.algo_params = algo_params
        method = getattr(self, algo_params["name"])
        denoiser_args['device'] = self.device
        step_size = algo_params.get('step_size', 1.9)

        stabilizer_default_args = {
            'relax': 1, 'mu': 0.95, 'step_size': 1.9, 'algo': 'FBS',
            'noise_factor': 2, 'path': None, 'rw': False, 'ne': False, 'name': 'CCD'
        }
        stabilizer_default_args.update(stabilizer_args)
        stabilizer_args = stabilizer_default_args

        print("using stabilizer:", stabilizer_id)
        self.stabilizer_path = stabilizer_args.get('path', None)

        if not dynamic_anchor:
            self.set_fixed_point(stabilizer_id, stabilizer_args)

        y = self.start_image.copy()
        yD = self.start_image.copy()
        yD2 = self.start_image.copy()
        y_old, yD_old, yD2_old = y.copy(), yD.copy(), yD2.copy()
        z = np.zeros_like(self.start_image)
        z_old = z.copy()

        N, ND, ND2 = [], [], []
        all_iters, all_itersD, all_itersD2 = [y.copy()], [yD.copy()], [yD2.copy()]

        psnr_value, ssim_value = self.get_metrics(y)
        psnrs, psnrsD, psnrsD2 = [psnr_value], [psnr_value], [psnr_value]
        ssims, ssimsD, ssimsD2 = [ssim_value], [ssim_value], [ssim_value]

        state = (y.copy(), y.copy(), 0)
        stateD = (yD.copy(), yD.copy(), 0)
        stateD2 = (yD2.copy(), yD2.copy(), 0)

        all_Thetas, all_unstb_etas, all_stb_etas, all_cfs = [], [], [], []
        vanilla_pnp_time, ours_pnp_time, equiv_pnp_time = 0, 0, 0
        vanilla_pnp_times, ours_pnp_times, equiv_pnp_times = [], [], []

        for i in range(num_iterations):
            # --- Ours: PnP + Stabilizer ---
            ours_start = time.time()
            x = method(state, step_size, denoiser, denoiser_args, denoiser_object)[0].copy()

            if not dynamic_anchor:
                eta_unst = np.linalg.norm(x.ravel() - self.stablerFP.ravel()) / \
                    np.linalg.norm(y.ravel() - self.stablerFP.ravel())
                z = self.stabilizer(state[0], stabilizer_id, stabilizer_args)
                mu = np.linalg.norm(z.ravel() - self.stablerFP.ravel()) / \
                    np.linalg.norm(y.ravel() - self.stablerFP.ravel()) if dynamic_cf else stabilizer_args['mu']
            else:
                z_new = self.stabilizer(z, stabilizer_id, stabilizer_args)
                sy = self.stabilizer(y, stabilizer_id, stabilizer_args)
                eta_unst = np.linalg.norm(x.ravel() - z_new.ravel()) / \
                    np.linalg.norm(y.ravel() - z.ravel())
                mu = np.linalg.norm(sy.ravel() - z_new.ravel()) / \
                    np.linalg.norm(y.ravel() - z.ravel()) if dynamic_cf else stabilizer_args['mu']

            all_unstb_etas.append(eta_unst)
            all_cfs.append(mu)

            Theta_k = self.calculate_theta(x, z, y, eta_unst) if i > stabilizer_args['relax'] else 0.0
            y = (1 - Theta_k) * x + Theta_k * z

            ours_pnp_times.append(time.time() - ours_start)
            ours_pnp_time += ours_pnp_times[-1]
            state = (y.copy(), y.copy(), y.copy())
            all_Thetas.append(Theta_k)

            if not dynamic_anchor:
                eta_stb = np.linalg.norm(y.ravel() - self.stablerFP.ravel()) / \
                    np.linalg.norm(y_old.ravel() - self.stablerFP.ravel())
            else:
                eta_stb = np.linalg.norm(y.ravel() - z.ravel()) / \
                    np.linalg.norm(y_old.ravel() - z_old.ravel())
            all_stb_etas.append(eta_stb)

            if i % plot_interval == 0:
                print(f"Iter {i}: Theta={Theta_k:.4f}, eta_unst={eta_unst:.4f}, eta_stb={eta_stb:.4f}, mu={mu:.4f}")

            N.append(np.linalg.norm(y.ravel() - y_old.ravel()))
            psnr_value, ssim_value = self.get_metrics(y)
            psnrs.append(psnr_value)
            ssims.append(ssim_value)
            y_old = y.copy()
            z_old = z.copy()
            all_iters.append(y.copy())

            # --- Vanilla PnP ---
            vanilla_start = time.time()
            stateD = method(stateD, step_size, denoiser, denoiser_args, denoiser_object)
            yD = stateD[0].copy()
            vanilla_pnp_times.append(time.time() - vanilla_start)
            vanilla_pnp_time += vanilla_pnp_times[-1]

            psnr_value, ssim_value = self.get_metrics(yD)
            ND.append(np.linalg.norm(yD.ravel() - yD_old.ravel()))
            psnrsD.append(psnr_value)
            ssimsD.append(ssim_value)
            yD_old = yD.copy()
            all_itersD.append(yD.copy())

            # --- Equivariant PnP ---
            if equivariant:
                if equiv_object is None:
                    raise ValueError("Equivariant object not provided.")
                equiv_args = {**denoiser_args, 'random': random}
                equiv_start = time.time()
                stateD2 = method(stateD2, step_size, equiv_object, equiv_args, denoiser_object)
                yD2 = stateD2[0].copy()
                equiv_pnp_times.append(time.time() - equiv_start)
                equiv_pnp_time += equiv_pnp_times[-1]

                psnr_value, ssim_value = self.get_metrics(yD2)
                ND2.append(np.linalg.norm(yD2.ravel() - yD2_old.ravel()))
                psnrsD2.append(psnr_value)
                ssimsD2.append(ssim_value)
                yD2_old = yD2.copy()
                all_itersD2.append(yD2.copy())

            # --- Terminal Plots (animated, overwrites in-place) ---
            if plot_graphs and i % plot_interval == 0 and i > 1:
                plt_term.clear_terminal()
                plt_term.clear_figure()
                plt_term.subplots(3, 1)
                plt_term.subplot(1, 1)
                plt_term.plot(psnrs, label="Ours")
                plt_term.plot(psnrsD, label="Vanilla")
                if equivariant:
                    plt_term.plot(psnrsD2, label="Equivariant")
                plt_term.title("PSNRs")
                plt_term.xlabel("Iteration")
                plt_term.ylabel("PSNR (dB)")
                plt_term.theme("clear")
                plt_term.xscale('log')

                plt_term.subplot(2, 1)
                plt_term.plot(all_unstb_etas, label="eta_unstable")
                plt_term.plot(all_stb_etas, label="eta_stable")
                plt_term.title("Eta (Unstable vs Stable)")
                plt_term.xlabel("Iteration")
                plt_term.ylabel("Eta")
                plt_term.theme("clear")
                plt_term.xscale('log')

                plt_term.subplot(3, 1)
                plt_term.plot(all_Thetas, label="Theta_k")
                plt_term.title("Theta_k")
                plt_term.xlabel("Iteration")
                plt_term.ylabel("Theta")
                plt_term.theme("clear")
                plt_term.xscale('log')

                plt_term.show()
                print(f"Iter {i} | Ours: {psnrs[-1]:.2f} dB | Vanilla: {psnrsD[-1]:.2f} dB | Equiv: {psnrsD2[-1]:.2f} dB")
                print(f"Avg time — Vanilla: {vanilla_pnp_time/i:.4f}s | Equiv: {equiv_pnp_time/i:.4f}s | Ours: {ours_pnp_time/i:.4f}s")

        print(f"Final avg iter time — Vanilla: {vanilla_pnp_time/num_iterations:.4f}s | Equiv: {equiv_pnp_time/num_iterations:.4f}s | Ours: {ours_pnp_time/num_iterations:.4f}s")

        self.reconstruction = y.copy()
        self.reconstructionVanilla = yD.copy()
        self.reconstructionEquiv = yD2.copy()
        self.get_images(save=True)
