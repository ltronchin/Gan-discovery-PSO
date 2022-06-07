"""
Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
import argparse
import yaml
import time
from tqdm import tqdm
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.utils import util_general
from src.utils import util_data
from inverter.utils_ae import util_inverter_statistics as util_inverter
from src.inverter.utils_ae import util_report_inverter

def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=str, default=None)
    parser.add_argument("-d", "--device_type", help="Select CPU or GPU", type=str, default="gpu")
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("-s", "--dataset", help="Dataset to upload", type=str, default="mnist")
    parser.add_argument("-m", "--model", help="Model of GAN to use", type=str, default="dcgan")
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()

# Configuration file
debug = 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
print("\nUpload configuration file")
if debug == 'develop':
    with open('./configs/dcgan_mnist.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    device_type = cfg['device']['device_type']
    worker = cfg['device']['worker']
    dataset_name = cfg['data']['dataset']
    model_gan = cfg['model_inverter']['model_name']
else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    device_type = args.device_type # todo notused!
    worker = args.gpu
    dataset_name = args.dataset
    model_gan = args.model

# SUBMIT RUN:
print("\nSubmit run")
# - Create log dir
# - Get new id_exp
# - Save the configuration file
# - Initialize Logger
# - Copy the code
run_module = os.path.basename(__file__)
if id_exp is None:
    run_id = util_general.get_next_run_id_local(os.path.join('log_run', dataset_name), run_module) # GET run id
else:
    run_id = id_exp
run_name = "{0:05d}--{1}".format(run_id, run_module)
log_dir = os.path.join('log_run', dataset_name, run_name)
util_general.create_dir(log_dir)
with open(os.path.join(log_dir, 'configuration.yaml'), 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
logger = util_general.Logger(file_name=os.path.join(log_dir, 'log.txt'), file_mode="w", should_flush=True)
files = util_general.list_dir_recursively_with_ignore('src', ignores=['.DS_Store', 'models'], add_base_to_relative=True)
files = [(f[0], os.path.join(log_dir, f[1])) for f in files]
util_general.copy_files_and_create_dirs(files)
print(f'Configuration args: {id_exp, worker, device_type, dataset_name, model_gan}')

# Seed everything
print("\nSeed all")
util_general.seed_all(cfg['seed'])

# Parameters
print("\nParameters")
image_size = cfg['data']['image_size']
channel = cfg['data']['channel']
iid_class = cfg['data']['iid_classes']
ood_class  = cfg['data']['ood_classes']
print(f'Parameters: {image_size, channel, iid_class, ood_class}')

# Register and history
print("\nInitialize history")
overall_time = util_general.nested_dict()
overall_history = {}
start = time.time()

# Device
print("\nSelect device")
device = torch.device(worker if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(f'device: {device}')

# Files and Directories
print('\nCreate file and directory')
data_dir = os.path.join(cfg['data']['data_dir'], dataset_name)
interim_dir = os.path.join(cfg['data']['interim_dir'], dataset_name, run_name)
util_general.create_dir(interim_dir)
model_dir = os.path.join(cfg['data']['model_dir'], dataset_name, run_name)
util_general.create_dir(model_dir)
reports_dir = os.path.join(cfg['data']['reports_dir'], dataset_name, run_name)
util_general.create_dir(reports_dir)
plot_training_dir = os.path.join(reports_dir, "training_plot")
util_general.create_dir(plot_training_dir)
general_reports_dir = os.path.join(reports_dir, "general")
util_general.create_dir(general_reports_dir)

# Data loaders>upload images from ood distrribution
print("\nCreate dataloader")
ood_train_dataset, ood_val_dataset = util_data.get_public_dataset(dataset_name=dataset_name, data_dir=data_dir,  model_name='dcgan', general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=ood_class)
# Define the dataloader
loader = DataLoader(dataset=ood_train_dataset, batch_size=1, shuffle=True, drop_last=False)
print("Sanity check on data")
images, labels = iter(loader).next()
assert torch.min(images) == -1
assert torch.max(images) == 1
print(f"Min: {torch.min(images)}, Max: {torch.max(images)}")

# Architecture
print("\nModel architecture")
# Domain-Guided Encoder
print("Build Domain-Guided Encoder")
E = util_inverter.Encoder(enc_dim=cfg['model_inverter']['latent_space'] ,channels_img=channel)
print("Load weights")
source_model_dir = util_general.define_source_path(cfg['data']['model_dir'], cfg['data']['dataset'], source_id_run=int('00001'), source_run_module='inverter.py')
E.load_state_dict(torch.load(os.path.join(source_model_dir, f"encoder.pt"), map_location=torch.device(device)))
E.to(device)
# Generator
print("Buil Generator")
G = util_inverter.Generator(z_dim=cfg['trainer_gan']['z_dim'], channels_img=channel, features_g=cfg['model_gan']['network']['units_gen'])
print("Load weights")
source_model_dir = util_general.define_source_path(cfg['data']['model_dir'], cfg['data']['dataset'], source_id_run=int('00001'), source_run_module='dcgan.py')
checkpoint_g = torch.load(os.path.join(source_model_dir, 'checkpoint_g.tar'),   map_location=torch.device(device))
G.load_state_dict(checkpoint_g['model_state_dict'])
G.to(device)

print("\nStart inversion")
tik = time.time()
latent_codes = []
# Invertion>Invert single image at time
idx_batch = 0

source_interim_dir = util_general.define_source_path(cfg['data']['interim_dir'], cfg['data']['dataset'],  source_id_run=int('00001'), source_run_module='pso_discovery.py')
with tqdm(total=len(loader.dataset), unit='img') as pbar:
    for ori_img, ori_label in loader:
        assert torch.min(ori_img) >= -1
        assert torch.max(ori_img) <= 1

        code, results, results_vis, log_message = util_inverter.invert_bn(general_reports_dir=general_reports_dir,
                                                                          plot_training_dir=plot_training_dir,
                                                                          source_interim_dir=source_interim_dir,
                                                                          iid_classes=iid_class,
                                                                          x=ori_img,
                                                                          generator=G,
                                                                          encoder=E,
                                                                          device=device,
                                                                          early_stopping=50)


        latent_batch = code[0].squeeze(-1).squeeze(-1)
        latent_batch = np.concatenate((latent_batch, util_inverter.get_tensor_value(ori_label)), axis=0)
        if idx_batch == 0:
            latent_ood = latent_batch
        else:
            latent_ood =  np.vstack((latent_ood, latent_batch))

        util_report_inverter.save_image(os.path.join(general_reports_dir, 'ori.png'), util_inverter.postprocess(util_inverter.get_tensor_value(ori_img))[0][0])
        util_report_inverter.save_image(os.path.join(general_reports_dir, 'enc.png'), results_vis[1][0])
        util_report_inverter.save_image(os.path.join(general_reports_dir, 'inv.png'), results_vis[-1][0])
        idx_batch += 1

        pbar.set_description_str(log_message)
        pbar.update(ori_img.shape[0])

        #if idx_batch >= 100:
        #    break
tok = time.time()

# Dataframe: ood_class
# row -> n_images of ood_class
# column -> feature
df = pd.DataFrame(latent_ood)
df.iloc[:, -1] = df.iloc[:, -1].astype('uint8')
with open(os.path.join(interim_dir, f'particles_position_ood.pkl'), 'wb') as f:
    pickle.dump(df, f)

print('\nDefining the run')
# Defining the run: save time and history
stop = time.time()
overall_time['training_time'] = (tok - tik)//len(loader.dataset)
overall_time['overall_time'] = stop - start
with open(os.path.join(general_reports_dir, f'timing.pkl'), 'wb') as f:
    pickle.dump(overall_time, f)
with open(os.path.join(general_reports_dir, f'overall_history.pkl'), 'wb') as f:
    pickle.dump(overall_history, f)

print(f"Total time: {util_general.format_time(stop-start)}")
util_general.notification_ifttt(info=f"May be the force with you! Time elapsed:{util_general.format_time(stop-start)}")
print("May be the force with you!")
