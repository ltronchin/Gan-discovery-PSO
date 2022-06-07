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
import torch.nn as nn

from src.utils import util_dcgan
from src.utils import util_general
from src.utils import util_data
from src.evaluation import util_cae, util_classifiers

# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--dataset", help="Dataset to upload", type=str, default="mnist")

    parser.add_argument("--latent_dim", help="Dimension of the GAN latent space", type=int, default=100)
    parser.add_argument("--path_den_cae", help="Path to the pretrained denoising CAE model", type=str, default="./models/mnist/00001--cae.py")
    parser.add_argument("--path_classifiers", help="Path to the pretrained classifiers", type=str, default="./models/mnist/00001--classifiers.py")

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
    den_cae_dir =  cfg['prerequisites']['model_den_cae']
    classifiers_dir =  cfg['prerequisites']['model_classifiers']
else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    id_exp = args.id_exp
    worker = args.gpu
    dataset_name = args.dataset

    latent_dim = args.latent_dim
    den_cae_dir =  args.path_den_cae
    classifiers_dir = args.path_classifiers

# Submit run:
print("Submit run")
run_module = os.path.basename(__file__)
# Get new id_exp
if id_exp is None:
    run_id = util_general.get_next_run_id_local(os.path.join('log_run', dataset_name), run_module) # GET run id
else:
    run_id = id_exp
# Create log dir
run_name = "{0:05d}--{1}".format(run_id, run_module)
log_dir = os.path.join('log_run', dataset_name, run_name)
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
print(f"cae_pretrained_dir: {den_cae_dir}")
print(f"classifiers_pretrained_dir: {classifiers_dir}")
print(f"iid classes:{iid_class}")
print(f"ood classes:{ood_class}")

# Register and history
print("Initialize history")
overall_time = util_general.nested_dict()
overall_history = {}
start = time.time()

# Device
print("Select device")
device = torch.device(worker if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(f'device: {device}')

# Files and Directories
print('Create file and directory')
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

# Data loaders
print("Create dataloader")
train_dataset, val_dataset = util_data.get_public_dataset(dataset_name=dataset_name, data_dir=data_dir, drange_net=cfg['data']['drange_net'], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=iid_class)
train_loader = DataLoader(train_dataset, batch_size=cfg["trainer_gan"]["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size = cfg['trainer_gan']['batch_size'], shuffle=True, drop_last=True)

# Check on the data
print("Sanity check on data")
for dset, loader in zip(['Training', 'Validation'], [train_loader, val_loader]):
    images, labels = iter(loader).next()
    print(f"{dset} Min: {torch.min(images)}, Max: {torch.max(images)}")

# Upload requirements for the run (pretrained models, data, ...)
print("Upload requirements")
print("Upload classifiers, encoder and decoder for evaluation phase")
classifiers = util_classifiers.load_classifiers(model_dir=classifiers_dir)
encoder, decoder = util_cae.load_autoencoder(model_dir=den_cae_dir, latent_space=cfg['model_ae']['latent_space'], device=device)

# Architecture
print("Model architecture")
# Sanity check!
util_dcgan.sanity_check(z_dim=latent_dim, device=device, n_img=8, image_size=cfg['data']['image_size'], channels_img=cfg['data']['channel'])
generator = util_dcgan.Generator(z_dim=latent_dim, channels_img=cfg['data']['channel'], features_g=cfg['model_gan']['network']['units_gen'])
generator.to(device)
discriminator = util_dcgan.Discriminator(channels_img=cfg['data']['channel'], features_d=cfg['model_gan']['network']['units_disc'])
discriminator.to(device)

util_dcgan.initialize_weights(generator)
util_dcgan.initialize_weights(discriminator)

# Optimizer and loss
print("Optimizer and loss")
opt_gen = util_dcgan.get_opti(generator.parameters(), **cfg['trainer_gan']['optimizer'])
opt_disc = util_dcgan.get_opti(discriminator.parameters(), **cfg['trainer_gan']['optimizer'])
criterion = nn.BCELoss().to(device) # Binary Cross Entropy between

# Start Training
print("Start training")
util_general.notification_ifttt(info="DCGAN Training started")
tik = time.time()
util_dcgan.train(
    general_reports_dir=general_reports_dir,
    plot_training_dir=plot_training_dir,
    model_dir=model_dir,
    epochs=cfg['trainer_gan']['epochs'],
    device=device,
    loader=train_loader,
    z_dim=latent_dim,
    batch_size=cfg['trainer_gan']['batch_size'],
    image_size=cfg['data']['image_size'],
    generator=generator,
    discriminator=discriminator,
    criterion=criterion,
    optimizer_gen=opt_gen,
    optimizer_disc=opt_disc,
    y_smoothing=cfg['trainer_gan']['label_smoothing'],
    resume_training = cfg['model_gan']['resume_training'],
    encoder = encoder,
    decoder = decoder,
    noise_factor = cfg['model_ae']['noise_factor'],
    classifiers = classifiers,
    val_loader = val_loader,
)
tok = time.time()

# Defining the run: save time and history
print('Defining the run')
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
