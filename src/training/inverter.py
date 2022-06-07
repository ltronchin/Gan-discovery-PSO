"""

"""
#  Libraries
print('Import the library')
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import argparse
import yaml
import time
import pickle

import torch
from torch.utils.data import DataLoader

from src.utils import util_general
from src.utils import util_data
from src.inverter.utils_ae import util_inverter
from src.pso import util_cnn

# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--dataset", help="Dataset to upload", type=str, default="mnist")

    parser.add_argument("--latent_dim", help="Dimension of the GAN latent space", type=int, default=100)
    parser.add_argument("--path_cnn", help="Path to the pretrained CNNs", type=str, default="./models/mnist/00001--cnn_multipatient.py")
    parser.add_argument("--path_gan", help="Path to the pretrained GAN", type=str, default="./models/mnist/00001--dcgan.py")

    parser.add_argument("--inverter_train_fun", help="Training function of inverter to select", type=str, default="pix_fea_rec_adv")

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()

# Configuration file
print("Upload configuration file")
debug = 'enter' # 'develop'
#debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
if debug == 'develop':
    with open('./configs/dcgan_mnist.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    worker = cfg['device']['worker']
    dataset_name = cfg['data']['dataset']

    latent_dim = cfg['trainer_gan']['z_dim']
    cnn_dir =  cfg['prerequisites']['model_cnn']
    gan_dir = cfg['prerequisites']['model_gan']

    training_fun = cfg['trainer_inverter']['training_function']

else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    dataset_name = args.dataset

    latent_dim = args.latent_dim
    cnn_dir =  args.path_cnn
    gan_dir = args.path_gan

    training_fun = args.inverter_train_fun

# Submit run:
print("Submit run")

new_root_dir=gan_dir.split(sep='/')[-1] # NEW ROOT DIR!

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
print(f"cnn pretrained dir: {cnn_dir}")
print(f"gan pretrained dir: {gan_dir}")
print(f"Training function for inverter: {training_fun}")
print(f"iid classes:{iid_class}")
print(f"ood classes:{ood_class}")

# Register and history
print("Initialize history")
overall_time = util_general.nested_dict()
overall_history = util_general.nested_dict()
start = time.time()

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
print("Create dataloader")
iid_train_dataset, iid_val_dataset = util_data.get_public_dataset_inverter(dataset_name=dataset_name, data_dir=data_dir, drange_net=cfg['data']['drange_net'], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=iid_class)
ood_train_dataset, ood_val_dataset = util_data.get_public_dataset_inverter(dataset_name=dataset_name, data_dir=data_dir, drange_net=cfg['data']['drange_net'], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=ood_class)
# Define the dataloader
train_loader = DataLoader(dataset=iid_train_dataset, batch_size=cfg['trainer_ae']['batch_size'], shuffle=True, drop_last=True)
valid_iid_loader = DataLoader(dataset=iid_val_dataset, batch_size=cfg['trainer_ae']['batch_size'], shuffle=False,  drop_last=True)
valid_ood_loader = DataLoader(dataset=ood_train_dataset, batch_size=cfg['trainer_ae']['batch_size'], shuffle=False,  drop_last=True)
data_loaders = {
    'train': train_loader,
    'val_iid': valid_iid_loader,
    'val_ood': valid_ood_loader
}

# Check on the data
print("Sanity check on data")
for dset, loader in zip(['Training', 'Validation_iid', 'Validation_ood'], [train_loader, valid_iid_loader, valid_ood_loader]):
    images, labels = iter(loader).next()
    print(f"{dset} Min: {torch.min(images)}, Max: {torch.max(images)}")

# Upload requirements for the run (pretrained models, data, ...)
print("Upload requirements")
print("Upload GAN and CNN")
# GAN
# Generator
print("Upload GAN generator")
checkpoint_g = torch.load(os.path.join(gan_dir, 'best_g.tar'), map_location=torch.device(device))
G = util_inverter.Generator(z_dim=latent_dim, channels_img=channel, features_g=cfg['model_gan']['network']['units_gen'])
G.load_state_dict(checkpoint_g['model_state_dict'])
G.to(device)

# CNN
print("Upload cnn")
try:
    if 'cnn_multipatient' in cnn_dir.split(sep='/')[-1]:
        print('Multipatient cnn')
        model_cnn = util_cnn.get_cnn(model_name=cfg['model_cnn']['model_name'], image_channels=channel,  iid_classes=iid_class, n_class=len(iid_class), img_dim=image_size, cnn_args=cfg['model_cnn']['network'])
        model_cnn.load_state_dict(torch.load(os.path.join(cnn_dir, f"model.pt"), map_location=torch.device(device)))
        model_cnn = model_cnn.to(device)
    else:
        raise NotImplementedError # todo
        # model_cnn_dict = {}
        # for label in iid_class:
        #     print(f"iid_class: {label}")
        #     model_cnn = util_cnn.get_cnn(model_name=cfg['model_cnn']['model_name'], image_channels=channel,  iid_classes=iid_class, n_class=2, img_dim=image_size, cnn_args=cfg['model_cnn']['network'])
        #     model_cnn.load_state_dict(torch.load(os.path.join(cnn_dir, f"model_{label}.pt"), map_location=torch.device(device)))
        #     model_cnn.to(device)
        #     model_cnn_dict[label] = model_cnn
except FileNotFoundError:
    print('CNN not found, skip')
    pass

# Architecture
print("Model architecture")
print("Build encoder")
E = util_inverter.Encoder(enc_dim=latent_dim, channels_img=channel)
E.to(device)
print("Build discriminator")
D = util_inverter.Discriminator(channels_img=channel, features_d=cfg['model_inverter']['D_network']['units_disc'])
D.to(device)
print("\nSanity check model")
try:
    util_inverter.sanity_check(z_dim=latent_dim, device=device, n_img=8, image_size=image_size, channels_img=channel)
except AttributeError:
    print('Error in summary function from torchsummary')
    pass

print("Initialization")
util_inverter.initialize_weights(E)
util_inverter.initialize_weights(D)

# Start Training
print("Start training")
util_general.notification_ifttt(info=f"Training inverter, latent space: {latent_dim}")
tik = time.time()
util_inverter.get_train_fun(training_fun=training_fun,
                            cfg=cfg,
                            general_reports_dir=general_reports_dir,
                            plot_training_dir= plot_training_dir,
                            model_dir=model_dir,
                            epochs=cfg['trainer_inverter']['epochs'],
                            data_loaders=data_loaders,
                            latent_dim=latent_dim,
                            encoder=E,
                            decoder=G,
                            device=device,
                            discriminator=D,
                            multipatient_cnn=model_cnn)
tok = time.time()

print('\nDefining the run')
# Defining the run: save time and history
stop = time.time()
overall_time['training_time'] = tok - tik
overall_time['overall_time'] = stop - start
with open(os.path.join(general_reports_dir, f'timing.pkl'), 'wb') as f:
    pickle.dump(overall_time, f)
with open(os.path.join(general_reports_dir, f'overall_history.pkl'), 'wb') as f:
    pickle.dump(overall_history, f)

print(f"Total time: {util_general.format_time(stop-start)}")
util_general.notification_ifttt(info=f"May be the force with you! Time elapsed:{util_general.format_time(stop-start)}")
print("May be the force with you!")
