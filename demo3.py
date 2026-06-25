from deepinv_denoiser import *
from classes.COSTA_PnP_class import *
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


image_id = 'im_062'
color_mode = 'RGB'
device = 'cuda:0'
image_path = f"images/{image_id}.png"
save_path = f"demo_logs/"

my_image = img_COSTA_PnP(
    image_path, forward_model_name='superresolution',
    forward_model_args={'scale_factor': 4, 'kernel_id': 2, 'device': device},
    noise_level=0.03, color_mode=color_mode, save_path=save_path, crop=True, crop_size=256)
my_image.get_images(save=True)

run_drunet, denoiser_drunet = get_denoiser('DRUNet', device=device)

my_image.COSTA_PnP(
    run_drunet,
    {'sigma': 8/255},
    denoiser_drunet,
    num_iterations=801,
    plot_graphs=True,
    plot_interval=20,
    stabilizer_args={
        'relax': 0, 'algo': 'DRS', 'step_size': 4,
        'noise_factor': 1, 'ne': False, 'rw': False,
        'path': 'pretrained/ccd_color.pth', 'name': 'ccd'},
    stabilizer_id='ccd',
    equivariant=True,
    equiv_object=EquivDen,
    random=True,
    algo_params={'transpose': True, 'name': 'HQS', 'step_size': 5, 'clip': False})
