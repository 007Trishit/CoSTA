from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from classes.blur_utils import *
from classes.utils_restoration import imread_uint, crop_center
import os
import cv2
import hdf5storage


class img_PnP:

    def __init__(self, image_path, forward_model_name, forward_model_args={},
                 noise_level=0.00, kernel_path=None, color_mode='RGB',
                 crop=False, crop_size=256, seed_val=7, save_path="./results/"):

        self.image_path = image_path
        self.color_mode = color_mode
        self.n_channels = 3 if color_mode == 'RGB' else 1

        self.image = imread_uint(image_path, n_channels=self.n_channels).astype(np.float32) / 255.0

        if crop:
            cs = crop_size if isinstance(crop_size, (list, tuple)) else [crop_size, crop_size]
            self.image = crop_center(self.image, cs[0], cs[1])

        valid_forward_models = ("deblurring", "superresolution")
        if forward_model_name not in valid_forward_models:
            raise ValueError(f"Only {valid_forward_models} are supported, got: {forward_model_name}")

        self.forward_model = forward_model_name
        self.forward_model_args = forward_model_args
        self.noise_level = noise_level
        self.seed_val = seed_val
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.init_image()

    def init_image(self):
        self.set_forward_model()

        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed_val)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.apply_forward_model()
        self.observed = self.observed + np.random.normal(0, self.noise_level, self.observed.shape)
        self.observed_tensor = torch.from_numpy(self.observed).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        self.set_start_image()
        self.reconstruction = None

    def initialize_prox_kernel(self, img):
        self.FB, self.FBC, self.F2B, self.FBFy = pre_calculate_prox(img, self.kernel_tensor, self.sf)

    def data_fidelity_prox_step(self, x, step_size):
        curr_img = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        y_ = prox_solution_L2(curr_img, self.FB, self.FBC, self.F2B, self.FBFy, step_size, self.sf)
        y_ = y_.cpu().squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.float32)
        return y_

    def set_forward_model(self):
        if self.forward_model == "superresolution":
            # Super-resolution uses the kernels_12 blur set (kernel_id 0-7);
            # the scale_factor controls the downsampling rate.
            kernels = hdf5storage.loadmat('images/kernels/kernels_12.mat')['kernels']
            self.kernel_id = self.forward_model_args['kernel_id']
            if not (0 <= self.kernel_id < 8):
                raise ValueError("kernel_id must be in [0, 8) for superresolution")
            self.kernel = kernels[0, self.kernel_id]
        else:
            # Deblurring: Levin09 motion kernels (0-7), 25x25 Gaussian (8), 9x9 box (9).
            kernels = hdf5storage.loadmat('images/kernels/Levin09.mat')['kernels']
            self.kernel_id = self.forward_model_args['kernel_id']
            if self.kernel_id < 8:
                self.kernel = kernels[0, self.kernel_id]
            elif self.kernel_id == 8:
                m, n = 12., 12.
                y, x = np.ogrid[-m:m+1, -n:n+1]
                h = np.exp(-(x*x + y*y) / (2.*1.6*1.6))
                h[h < np.finfo(h.dtype).eps * h.max()] = 0
                self.kernel = h / h.sum()
            elif self.kernel_id == 9:
                self.kernel = (1/81) * np.ones((9, 9))
            else:
                raise ValueError("kernel_id must be < 10")

        self.sf = self.forward_model_args['scale_factor']
        self.device = self.forward_model_args['device']
        self.kernel_tensor = torch.from_numpy(self.kernel).float().to(self.device)
        self.A_function = G
        self.A_function_adjoint = Gt
        self.A_kwargs = {'k': self.kernel_tensor, 'sf': self.sf}
        self.A_adjoint_kwargs = {'k': self.kernel_tensor, 'sf': self.sf}
        self.op_norm = get_op_norm(self.A_kwargs, self.A_adjoint_kwargs, self.n_channels,
                                   self.device, img_size=(self.image.shape[0], self.image.shape[1]))
        print(f"Operator norm: {self.op_norm}, step size <= {2/self.op_norm**2:.4f}")

    def apply_forward_model(self):
        img = torch.from_numpy(self.image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.observed = self.A_function(img, **self.A_kwargs).cpu().squeeze(0).permute(1, 2, 0).detach().numpy()

    def set_start_image(self):
        if self.forward_model == "superresolution" and self.sf > 1:
            # Initialize SR from a bicubic upsampling of the low-res observation.
            h, w = self.image.shape[0], self.image.shape[1]
            start = cv2.resize(self.observed, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            if start.ndim == 2:
                start = start[:, :, None]
            self.start_image = start
        else:
            self.start_image = self.observed.copy()
        self.initialize_prox_kernel(self.observed_tensor)

    def get_metrics(self, my_image):
        try:
            my_psnr = psnr(self.image, my_image, data_range=1.0)
            if self.color_mode == 'L':
                my_ssim = ssim(self.image[:, :, 0], my_image[:, :, 0], data_range=1.0)
            else:
                my_ssim = np.mean([ssim(self.image[:, :, c], my_image[:, :, c], data_range=1.0) for c in range(3)])
        except:
            my_psnr, my_ssim = None, None
        return my_psnr, my_ssim

    def get_images(self, save=False):
        if save and self.save_path:
            self._save_image(self.image, "original_image.png")
            self._save_image(self.observed, "observed_image.png")
            self._save_image(self.start_image, "start_image.png")
            if self.reconstruction is not None:
                self._save_image(self.reconstruction, "reconstructed_image.png")

    def _save_image(self, image, name):
        img = np.clip(image * 255, 0, 255).astype(np.uint8)
        if self.color_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.save_path + name, img)

    def save_any_image(self, image, name):
        self._save_image(image, name)

    def get_Gradient(self, y):
        y_tensor = torch.from_numpy(y).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        grad = self.A_function_adjoint(
            self.A_function(y_tensor, **self.A_kwargs) - self.observed_tensor,
            **self.A_adjoint_kwargs
        ).cpu().squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.float32)
        return grad

    def grad_desc_step(self, y, step_size):
        return y - (step_size / self.op_norm ** 2) * self.get_Gradient(y)

    def FBS(self, y, step_size, denoiser, denoiser_args, denoiser_object):
        y = y[0]
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']

        y_temp = self.grad_desc_step(y, step_size).astype(np.float32)
        x = denoiser(y_temp.transpose(2, 0, 1), denoiser_object, **denoiser_args).transpose(1, 2, 0) \
            if transpose else denoiser(y_temp, denoiser_object, **denoiser_args)
        if clip:
            x = np.clip(x, 0, 1)
        return (x.copy(), 0, 0)

    def HQS(self, y, step_size, denoiser, denoiser_args, denoiser_object):
        y = y[0]
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']

        y_temp = self.data_fidelity_prox_step(y, step_size).astype(np.float32)
        x = denoiser(y_temp.transpose(2, 0, 1), denoiser_object, **denoiser_args).transpose(1, 2, 0) \
            if transpose else denoiser(y_temp, denoiser_object, **denoiser_args)
        if clip:
            x = np.clip(x, 0, 1)
        return (x.copy(), 0, 0)

    def RED(self, y, step_size, denoiser, denoiser_args, denoiser_object):
        y = y[0]
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']
        lam = self.algo_params.get('lambda', 1.0)

        Dy = denoiser(y.transpose(2, 0, 1), denoiser_object, **denoiser_args).transpose(1, 2, 0) \
            if transpose else denoiser(y, denoiser_object, **denoiser_args)
        x = self.grad_desc_step(y, step_size) - step_size * lam * (y - Dy)
        if clip:
            x = np.clip(x, 0, 1)
        return (x.copy(), 0, 0)

    def DRS(self, y, step_size, denoiser, denoiser_args, denoiser_object):
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']

        u, v, b = y

        v = self.data_fidelity_prox_step(u, step_size)

        # applying denoiser
        b = denoiser((2*v - u).transpose(2, 0, 1),
                     denoiser_object, **denoiser_args).transpose(1, 2, 0) if transpose else denoiser(2*v - u, denoiser_object, **denoiser_args)

        u = u + (b - v)

        if clip:
            v = np.clip(v, 0, 1)
            u = np.clip(u, 0, 1)
            b = np.clip(b, 0, 1)

        return (u.copy(), v.copy(), b.copy())
