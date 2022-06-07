"""

"""
#  Libraries
print('Import the library')
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])


from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
import os
import yaml
import pickle

import torch
from torch.utils.data import DataLoader

from src.utils import util_general
from src.utils import util_data
from src.inverter.utils_ae import util_inverter
from src.inverter.utils_ae import util_report_inverter

# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--dataset", help="Dataset to upload", type=str, default="mnist")

    parser.add_argument("--latent_dim", help="Dimension of the GAN latent space", type=int, default=100)
    parser.add_argument("--path_inverter", help="Path to the pretrained Inverter", type=str, default="./models/mnist/00001--inverter. py")
    parser.add_argument("--path_gan", help="Path to the pretrained GAN", type=str, default="./models/mnist/00001--dcgan.py")

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()

# Configuration file
print("Upload configuration file")
debug = 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
if debug == 'develop':
    with open('./configs/dcgan_mnist.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    worker = cfg['device']['worker']
    dataset_name = cfg['data']['dataset']

    latent_dim = cfg['trainer_gan']['z_dim']
    inverter_dir =  cfg['prerequisites']['model_inverter']
    gan_dir = cfg['prerequisites']['model_gan']

else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    dataset_name = args.dataset

    latent_dim = args.latent_dim
    inverter_dir =  args.path_inverter
    gan_dir = args.path_gan

# Submit run:
print("Submit run")

new_root_dir_gan=gan_dir.split(sep='/')[-1] # NEW ROOT DIR!
new_root_dir_inverter = inverter_dir.split('/')[-1]
new_root_dir = os.path.join(new_root_dir_gan, new_root_dir_inverter)
inverter_dir = os.path.join(gan_dir, inverter_dir.split('/')[-1])

run_module = os.path.basename(__file__)
# Get new id_exp
if id_exp is None:
    run_id = util_general.get_next_run_id_local(os.path.join('log_run', dataset_name, new_root_dir), run_module) # GET run id
else:
    run_id = id_exp
# Create log dir
run_name = "{0:05d}--{1}".format(run_id, run_module)
log_dir = os.path.join('log_run', dataset_name, new_root_dir, run_name)
util_general.create_dir(log_dir)
# Save the configuration file
with open(os.path.join(log_dir, 'configuration.yaml'), 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
# Initialize Logger
logger = util_general.Logger(file_name=os.path.join(log_dir, 'log.txt'), file_mode="w", should_flush=True)
# Copy the code in log_dir
files = util_general.list_dir_recursively_with_ignore('src', ignores=['.DS_Store', 'models'], add_base_to_relative=True)
files = [(f[0], os.path.join(log_dir, f[1])) for f in files]
util_general.copy_files_and_create_dirs(files)

# Welcome
from datetime import datetime
now = datetime.now()
date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
print("Hello!",date_time)

# Seed everything
print("Seed all")
util_general.seed_all(cfg['seed'])

# Parameters
print("Parameters")
image_size = cfg['data']['image_size']
channel = cfg['data']['channel']
iid_class = cfg['data']['iid_classes']
ood_class  = cfg['data']['ood_classes']

# Useful print
print(f"id_exp: {id_exp}")
print(f"Latent_dim: {latent_dim}")
print(f"inverter pretrained dir: {inverter_dir}")
print(f"gan pretrained dir: {gan_dir}")
print(f"iid classes:{iid_class}")
print(f"ood classes:{ood_class}")

# Register and history
print("Initialize history")

# Device
print("Select device")
device = torch.device(worker if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(f'device: {device}')

# Files and Directories
print('Create file and directory')
data_dir = os.path.join(cfg['data']['data_dir'], dataset_name)

interim_dir = os.path.join(cfg['data']['interim_dir'], dataset_name, new_root_dir, run_name)
util_general.create_dir(interim_dir)
model_dir = os.path.join(cfg['data']['model_dir'], dataset_name, new_root_dir, run_name)
util_general.create_dir(model_dir)
reports_dir = os.path.join(cfg['data']['reports_dir'], dataset_name, new_root_dir, run_name)
util_general.create_dir(reports_dir)
plot_training_dir = os.path.join(reports_dir, "training_plot")
util_general.create_dir(plot_training_dir)
general_reports_dir = os.path.join(reports_dir, "general")
util_general.create_dir(general_reports_dir)

# Data loaders
print("Create ood dataloader")
ood_train_dataset, _ = util_data.get_public_dataset_inverter(dataset_name=dataset_name, data_dir=data_dir, drange_net=cfg['data']['drange_net'], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=ood_class)
# Define the dataloader
ood_data_loader = DataLoader(dataset=ood_train_dataset, batch_size=cfg['trainer_ae']['batch_size'], shuffle=True, drop_last=False)
# Check on the data
print("Sanity check on data")
for dset, loader in zip(['ood'], [ood_data_loader]):
    images, labels = iter(loader).next()
    print(f"{dset} Min: {torch.min(images)}, Max: {torch.max(images)}")

# Upload requirements for the run (pretrained models, data, ...)
print("Upload requirements")
print("Upload GAN and Inverter")
# GAN
# Generator
print("Upload GAN generator")
checkpoint_g = torch.load(os.path.join(gan_dir, 'best_g.tar'), map_location=torch.device(device))
G = util_inverter.Generator(z_dim=latent_dim, channels_img=channel, features_g=cfg['model_gan']['network']['units_gen'])
G.load_state_dict(checkpoint_g['model_state_dict'])
G.to(device)
# Inverter
print("Upload Inverter (Encoder)")
E = util_inverter.Encoder(enc_dim=latent_dim, channels_img=channel)
E.load_state_dict(torch.load(os.path.join(inverter_dir, f"encoder.pt"), map_location=torch.device(device)))
E.to(device)

E.eval()
G.eval()
print("nStart extraction")
for label in ood_class:
    print(f"ood_class:{label}")

    # Dir
    general_reports_dir_label = os.path.join(general_reports_dir, str(label))
    util_general.create_dir(general_reports_dir_label)

    idx_batch = 0
    with tqdm(total=len(ood_train_dataset), unit='img') as pbar:
        for x_batch, y_batch in ood_data_loader:

            assert x_batch.dtype == torch.float32
            assert torch.max(x_batch) <= 1.0
            assert torch.min(x_batch) >= -1.0
            assert x_batch[0].shape[0] == 1

            y_mask = y_batch == label # select only the ood_class of interest
            x_batch = x_batch[y_mask].to(device)
            y_batch = y_batch[y_mask].to(device)
            with torch.no_grad():
                latent_batch = E(x_batch.float())

            util_report_inverter.show_gan_images(general_reports_dir=general_reports_dir_label, epoch=idx_batch, noise=latent_batch, decoder=G)
            latent_batch = latent_batch.squeeze(dim=-1).squeeze(dim=-1)
            latent_batch = latent_batch.detach().cpu().numpy()

            if idx_batch == 0:
                latent_ood_class = latent_batch
            else:
                latent_ood_class = np.concatenate((latent_ood_class, latent_batch), axis=0)

            pbar.update(x_batch.shape[0])
            idx_batch += 1

    # Dataframe: ood_class
    # row -> n_images of ood_class
    # column -> feature
    df = pd.DataFrame(latent_ood_class)
    with open(os.path.join(interim_dir, f'particles_position_ood_class_{label}.pkl'), 'wb') as f:
        pickle.dump(df, f)
print("May be the force with you!")
