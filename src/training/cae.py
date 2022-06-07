# -----------------------------------
# Author
#   Lorenzo Tronchin
# Script description
#  Training of Convolutional Autoencoder for evaluation purpose
# Date
#
# -----------------------------------

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

from src.evaluation import util_cae
from src.utils import util_general
from src.utils import util_report
from src.utils import util_data

# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--dataset", help="Dataset to upload", type=str, default="mnist")

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
else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    dataset_name = args.dataset

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
image_size = cfg['data']['image_size']
channel = cfg['data']['channel']
iid_class = cfg['data']['iid_classes']
ood_class = cfg['data']['ood_classes']
task = cfg['model_ae']['task']

# Useful print
print(f"id_exp: {id_exp}")
print(f"cae_task: {task}")
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
train_dataset, val_dataset = util_data.get_public_dataset(dataset_name=dataset_name, data_dir=data_dir, drange_net=[0, 1], general_reports_dir=general_reports_dir,  image_size=image_size, channel=channel, iid_class=iid_class)
# Define the dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size = cfg['trainer_ae']['batch_size'], shuffle=True,  drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size = cfg['trainer_ae']['batch_size'], shuffle=False,  drop_last=True)

# Check on the data
print("Sanity check on data")
for dset, loader in zip(['Training', 'Validation'], [train_loader, val_loader]):
    images, labels = iter(loader).next()
    print(f"{dset} Min: {torch.min(images)}, Max: {torch.max(images)}")

# Upload requirements for the run (pretrained models, data, ...)

# Architecture
print("Model architecture")
# Initialize the two networks
encoder = util_cae.Encoder(encoded_space_dim=cfg['model_ae']['latent_space'])
decoder = util_cae.Decoder(encoded_space_dim=cfg['model_ae']['latent_space'])
if cfg['model_ae']['resume_training']:
    print("Upload pretrained model")
    encoder.load_state_dict(torch.load(os.path.join(cfg['model_ae']['resume_training'], dataset_name, "encoder.pt"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(cfg['model_ae']['resume_training'], dataset_name, "decoder.pt"), map_location=device))

# Optimizer and loss
print("Optimizer and loss")
loss = torch.nn.MSELoss() # Define the loss function
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optim = torch.optim.Adam(params_to_optimize, lr=cfg['trainer_ae']['optimizer']['lr'],  weight_decay=1e-05) # Define an optimizer
encoder = encoder.to(device)
decoder = decoder.to(device)

# Start Training
print("Start training")
util_general.notification_ifttt(info="CAE Training started")

tik = time.time()

history = {'train_loss': [], 'val_loss': []}
for epoch in range(cfg['trainer_ae']['epochs']):

    train_loss, encoder, decoder = util_cae.get_training_loop(
        task=task,
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=train_loader,
        loss_fn=loss,
        optimizer=optim,
        noise_factor=cfg['model_ae']['noise_factor']
    )

    val_loss, encoder, decoder = util_cae.get_test_loop(
        task=task,
        general_reports_dir=general_reports_dir,
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=val_loader,
        dataset=val_dataset,
        loss_fn=loss,
        noise_factor=cfg['model_ae']['noise_factor']
    )

    # Print Validation loss
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, cfg['trainer_ae']['epochs'], train_loss, val_loss))
    # Save model
    torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(model_dir, "decoder.pt"))
tok = time.time()
util_general.notification_ifttt(info="CAE Training stop")


util_report.plot_training(history=history, plot_training_dir=plot_training_dir)
# Create encoded samples for training set
encoded_samples_train = util_cae.create_encoded_sample(encoder=encoder, data=train_dataset, device=device)
encoded_samples_train.to_csv(os.path.join(interim_dir, "encoded_samples_train.csv")) # save the train embeddings
# Create encoded sample for validation set
encoded_samples_valid = util_cae.create_encoded_sample(encoder=encoder, data=val_dataset, device=device)
encoded_samples_valid.to_csv(os.path.join(interim_dir, "encoded_samples_valid.csv")) # save the validation embeddings
if cfg['model_ae']['latent_space'] == 2:
    util_cae.plot_img_latent_space(decoder=decoder, device=device, general_reports_dir=general_reports_dir, n=10, w=cfg['data']['image_size'])
    util_cae.plot_feature_latent_space(general_reports_dir=general_reports_dir, encoded_samples=encoded_samples_train, dataset='Training')
    util_cae.plot_feature_latent_space(general_reports_dir=general_reports_dir, encoded_samples=encoded_samples_valid, dataset='Validation')

# Defining the run: save time and history
print('Defining the run')
stop = time.time()
overall_time['training_time'] = tok - tik
overall_time['overall_time'] = stop - start
with open(os.path.join(general_reports_dir, f'timing.pkl'), 'wb') as f:
    pickle.dump(overall_time, f)

print(f"Total time: {util_general.format_time(stop-start)}")
util_general.notification_ifttt(info=f"May be the force with you! Time elapsed:{util_general.format_time(stop-start)}")
print("May be the force with you!")

