import warnings
warnings.filterwarnings("ignore")
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from classes.OursStabilizingMethod_PnP_class import *
from deepinv_denoiser import *

image_id = 'leaves'
color_mode = 'RGB'
device = 'cuda:0'
image_path = f"images/{image_id}.png"
save_path = f"demo_logs/"

my_image = img_OursStabilizingMethod_PnP(
    image_path, forward_model_name='deblurring',
    forward_model_args={'scale_factor': 1, 'kernel_id': 3, 'device': device},
    noise_level=0.03, color_mode=color_mode, save_path=save_path, crop=False, crop_size=256)
my_image.get_images(save=True)

run_drunet, denoiser_drunet = get_denoiser('DRUNet', device=device)

my_image.OursStabilizingMethod_PnP(
    run_drunet,
    {'sigma': 5/255},
    denoiser_drunet,
    num_iterations=1001,
    plot_graphs=True,
    plot_interval=10,
    stabilizer_args={
        'relax': 0, 'algo': 'HQS', 'step_size': 5,
        'noise_factor': 1, 'ne': False, 'rw': False,
        'path': 'pretrained/ccd_color.pth', 'name': 'ccd'},
    stabilizer_id='ccd',
    equivariant=True,
    equiv_object=EquivDen,
    random=True,
    algo_params={'transpose': True, 'name': 'FBS', 'step_size': 2.05, 'clip': False})
