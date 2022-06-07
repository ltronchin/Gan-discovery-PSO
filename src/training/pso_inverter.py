# Input
# Pretrained Inverter (Encoder)  -- freezed
# Pretrained Decoder (Generator) -- freezed
# ONE ood patient
# IID data
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
import time

import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from src.pso import util_cnn
from src.utils import util_general
from src.utils import util_data
from src.inverter.utils_ae import util_inverter
from src.inverter.utils_ae import util_pso_inverter
from src.pso import util_discovery
from src.utils import util_report
from src.pso import util_pso

# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--dataset", help="Dataset to upload", type=str, default="mnist")

    parser.add_argument("--latent_dim", help="Dimension of the GAN latent space", type=int, default=100)
    parser.add_argument("--path_cnn", help="Path to the pretrained CNN/CNNs", type=str,    default="./models/mnist/00001--cnn_multipatient.py")
    parser.add_argument("--path_gan", help="Path to the pretrained GAN", type=str, default="./models/mnist/00001--dcgan.py")
    parser.add_argument("--path_inverter", help="Path to the pretrained Inverter", type=str, default="./models/mnist/00001--inverter. py")

    #parser.add_argument("--path_ood_patient", help="Path to the folder of OoD patient", type=str, default="./")# todo add argument to select a path to a folder with the slice of a ood patient
    parser.add_argument("--path_ood_patient", help="ID of the OoD patient to select", type=int, default=5)

    parser.add_argument("--w_ine", help="Inertia coefficient of particle", type=float, default=0.73)
    parser.add_argument("--w_cogn", help="Cognitive coefficient of particle", type=float, default=1.496)
    parser.add_argument("--w_soci", help="Social coefficient of particle", type=float, default=1.496)
    parser.add_argument("--schedule_ine", help="Activate the schedule over iterations of inertia", type=bool,  default=False)
    parser.add_argument("--control_pso_fitness", help="If discover latent position near IiD (set it to True) or near OoD (set it to False)", type=str, default='optimize_in_training')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()

# Configuration file
print("Upload configuration file")
debug = 'enter'
#debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
if debug == 'develop':
    with open('./configs/dcgan_mnist.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    worker = cfg['device']['worker']
    dataset_name = cfg['data']['dataset']

    latent_dim = cfg['trainer_gan']['z_dim']
    cnn_dir = cfg['prerequisites']['model_cnn']
    gan_dir = cfg['prerequisites']['model_gan']
    inverter_dir =  cfg['prerequisites']['model_inverter']

    ood_patient = cfg['pso_inverter']['ood_patient']     # todo add folder to ood patient slice (path_ood_patient)

    w_ine = cfg['trainer_pso_inverter']['w_inertia']
    w_cogn = cfg['trainer_pso_inverter']['w_cognitive']
    w_soci = cfg['trainer_pso_inverter']['w_social']
    schedule_ine = cfg['trainer_pso_inverter']['schedule_inertia']
    control_pso_fitness = cfg['trainer_pso_inverter']['control_pso_fitness']
else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    dataset_name = args.dataset

    latent_dim = args.latent_dim
    cnn_dir = args.path_cnn
    gan_dir = args.path_gan
    inverter_dir =  args.path_inverter

    ood_patient = args.path_ood_patient     # todo add folder to ood patient slice

    w_ine = args.w_ine
    w_cogn =  args.w_cogn
    w_soci =  args.w_soci
    schedule_ine = args.schedule_ine
    control_pso_fitness = args.control_pso_fitness

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
    id_exp = run_id
else:
    run_id = id_exp
# Create log dir
run_name = "{0:05d}--{1}".format(run_id, run_module)
log_dir = os.path.join('log_run', dataset_name, new_root_dir, run_name, control_pso_fitness)
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

# Useful print
print(f"id_exp: {id_exp}")
print(f"Root_dir: {new_root_dir}")
print(f"Latent_dim: {latent_dim}")
print(f"iid classes:{iid_class}")

print(f"cnn_pretrained_dir: {cnn_dir}")
print(f"gan_pretrained_dir: {gan_dir}")
print(f"inverter pretrained dir: {inverter_dir}")

print(f"ood patient:{ood_patient}")

print(f"Inertia: {w_ine}")
print(f"Cognitive: {w_cogn}")
print(f"Social:{w_soci}")
print(f"Schedule inertia:{schedule_ine}")
print(f"Control PSO fitness:{control_pso_fitness}")

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

interim_dir = os.path.join(cfg['data']['interim_dir'], dataset_name, new_root_dir, run_name, control_pso_fitness)
util_general.create_dir(interim_dir)
model_dir = os.path.join(cfg['data']['model_dir'], dataset_name, new_root_dir, run_name, control_pso_fitness)
util_general.create_dir(model_dir)
reports_dir = os.path.join(cfg['data']['reports_dir'], dataset_name, new_root_dir, run_name, control_pso_fitness)
util_general.create_dir(reports_dir)
plot_training_dir = os.path.join(reports_dir, "training_plot")
util_general.create_dir(plot_training_dir)
general_reports_dir = os.path.join(reports_dir, "general")
util_general.create_dir(general_reports_dir)

# Data loaders CNN
print("Create dataloader for IID and OoD patient data")
iid_train_ood_patient_dataset, iid_val_ood_patient_dataset = util_data.get_public_dataset_inverter(dataset_name=dataset_name, data_dir=data_dir, drange_net=[0, 1], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=iid_class+[ood_patient])
train_loader = DataLoader(dataset=iid_train_ood_patient_dataset, batch_size=cfg['trainer_pso_inverter']['batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=iid_val_ood_patient_dataset, batch_size=cfg['trainer_pso_inverter']['batch_size'], shuffle=False,  drop_last=True)
data_loaders = {
        'train': train_loader,
        'val':val_loader
}
# Check on the data
print("Sanity check on data")
for dset, loader in zip(['iid_train_ood_patient_dataset', 'iid_val_ood_patient_dataset'], [train_loader, val_loader]):
    images, labels = iter(loader).next()
    print(f"{dset} Min: {torch.min(images)}, Max: {torch.max(images)}")
    assert torch.min(images) == 0.0
    assert torch.max(images) == 1.0

# Upload requirements for the run (pretrained models, data, ...)

print("Start PSO based inversion")

print("####")
print("Phase1: Assessor Fine Tuning on OoD patient data")
try:
    print(f'Try to upload Fine Tuned CNN for ood_patient {ood_patient}')
    model_cnn = util_cnn.get_cnn(model_name=cfg['model_pso_inverter']['model_name'], image_channels=channel, iid_classes=[0, 1], n_class=2, img_dim=image_size, cnn_args=cfg['model_cnn']['network'])
    model_cnn.load_state_dict(torch.load(os.path.join(model_dir, f"model_{ood_patient}.pt"), map_location=torch.device(device)))
    model_cnn = model_cnn.to(device) #todo check directory model_dir
    print('Success')
except FileNotFoundError:
    print(f"Start Fine Tuning CNN for ood_patient: {ood_patient}")

    print("Load pretrained CNN on iid Data")
    if 'cnn_multipatient' in cnn_dir.split(sep='/')[-1]:
        print('Multipatient cnn')
    model_cnn = util_cnn.get_cnn(model_name=cfg['model_cnn']['model_name'], image_channels=channel,  iid_classes=iid_class, n_class=len(iid_class), img_dim=image_size, cnn_args=cfg['model_cnn']['network'])
    model_cnn.load_state_dict(torch.load(os.path.join(cnn_dir, f"model.pt"), map_location=torch.device(device)))
    print(model_cnn)

    model_cnn = util_pso_inverter.change_classifier_n_class(model=model_cnn, n_class=2)
    model_cnn = model_cnn.to(device)
    # Train for a couple of epochs
    # Optimizer and loss
    print("Optimizer and loss")
    # Loss
    criterion = nn.CrossEntropyLoss().to(device)
    # Optimizer
    optimizer = util_cnn.get_opti(model_cnn.parameters(), **cfg['trainer_pso_inverter']['optimizer'])
    # LR Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['trainer_pso_inverter']['scheduler']['mode'],  patience=cfg['trainer_pso_inverter']['scheduler']['patience'])
    tik = time.time()
    util_general.notification_ifttt(info=f"CNN, ood_patient {ood_patient} Fine Tuning started")
    _, history_cnn = util_cnn.train_model(
        model=model_cnn, data_loaders=data_loaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        num_epochs=cfg['trainer_pso_inverter']['epochs'],
        early_stopping=cfg['trainer_pso_inverter']['early_stopping'], model_dir=model_dir, device=device, label=ood_patient
    )
    tok = time.time()
    print(f"Training time Binary Assessor: {tok - tik}")
    overall_time[f"cnn_time_ood_patient_{ood_patient}"]['training_time'] = tok - tik
    overall_history[f"cnn_history_ood_patient_{ood_patient}"] = history_cnn
    # Plot Training
    util_report.plot_training(history=history_cnn, plot_training_dir=plot_training_dir, label=ood_patient)

print("####")
print("Phase2: PSO based optimization")

# Data loaders
print("Create dataloader OoD patient") # todo dataloader solo sulle slice del corrente ood patient
ood_patient_dataset, _ = util_data.get_public_dataset_inverter(dataset_name=dataset_name, data_dir=data_dir, drange_net=cfg['data']['drange_net'], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=[ood_patient])
# Define the dataloader
ood_patient_loader = DataLoader(dataset=ood_patient_dataset, batch_size=1, shuffle=True, drop_last=False)
print("Sanity check on data")
for dset, loader in zip(['ood_patient'], [ood_patient_loader]):
    images, labels = iter(loader).next()
    print(f"{dset} Min: {torch.min(images)}, Max: {torch.max(images)}")
    assert torch.min(images) == -1.0
    assert torch.max(images) == 1.0
num_particles = len(ood_patient_dataset)
print(f"Number of particles for PSO: {num_particles} (Depends on the number of slices of the current patient)")
if num_particles > 256:
    print("Too much!")
    num_particles = 256
    print(f"Number of particles for PSO: {num_particles} (Depends on the number of slices of the current patient)")

# Upload requirements for the run (pretrained models, data, ...)
print("Upload requirements")
# Generator
print("Load pretrained GAN generator")
checkpoint_g = torch.load(os.path.join(gan_dir, 'best_g.tar'), map_location=torch.device(device))
G = util_inverter.Generator(z_dim=latent_dim, channels_img=channel, features_g=cfg['model_gan']['network']['units_gen'])
G.load_state_dict(checkpoint_g['model_state_dict'])
G=G.to(device)
# Inverter
print("Load pretrained Inverter (Encoder)")
E = util_inverter.Encoder(enc_dim=latent_dim, channels_img=channel)
E.load_state_dict(torch.load(os.path.join(inverter_dir, f"encoder.pt"), map_location=torch.device(device)))
E=E.to(device)

G = G.eval()
E = E.eval()

# PSO
print("Start PSO")
discovery = util_discovery.Discovery(iid_class=ood_patient, model_gan=G, model_cnn=model_cnn, device=device, control_pso_fitness=control_pso_fitness, obj_fun_threshold=0)

swarm = util_pso.Swarm(
    plot_training_dir=plot_training_dir,
    discovery=discovery,
    num_particles=num_particles,
    n_iterations=cfg['trainer_pso_inverter']['n_iterations'],
    dim_space=latent_dim,
    device=device,
    tolerance=cfg['trainer_pso_inverter']['tolerance'],
    w_inertia=w_ine,
    w_cogn=w_cogn,
    w_soci=w_soci
)

tik = time.time()
history_pso, history_particles, history_particles_vel, last_iteration = swarm.optimize(schedule_inertia=schedule_ine, early_stopping=cfg['trainer_pso_inverter']['early_stopping_pso'], ood_patient=ood_patient_loader, encoder=E)
tok = time.time()

# Plot Discovery
print("Plot training results")
util_report.plot_pso_convergence(plot_training_label_dir=general_reports_dir,  global_best_val=swarm.g_best_val_dummy)
util_report.plot_training(history=history_pso, plot_training_dir=general_reports_dir)
print("Plot latent space")
if latent_dim == 2:
    fitness_grid, img_grid = util_report.plot2d(plot_training_dir=plot_training_dir,  history_particles=history_particles, fitness=discovery.fitness,  g_best_pos=swarm.g_best_pos, grid_range=5, color='#000')
    with open(os.path.join(general_reports_dir, f'fitness_grid.pkl'), 'wb') as f:
        pickle.dump(fitness_grid, f)
    with open(os.path.join(general_reports_dir, f'img_grid.pkl'), 'wb') as f:
        pickle.dump(img_grid, f)
    util_report.make_gif_from_folder(plot_training_dir=plot_training_dir,    img_to_upload_dir=plot_training_dir, filename_result='2dspace_latent.gif',  filename_source='2d_plot')

util_report.plot_feature(plot_training_dir=plot_training_dir, history_particles=history_particles,   dim_space=latent_dim, iteration=last_iteration)
util_report.plot_features_last_iteration(plot_training_dir=plot_training_dir, history_particles=history_particles,  dim_space=latent_dim)
print("Make gif")
util_report.make_gif_from_folder(plot_training_dir=plot_training_dir, img_to_upload_dir=plot_training_dir)

with open(os.path.join(interim_dir, f'particles_position_ood_class_{ood_patient}.pkl'), 'wb') as f:
    pickle.dump(history_particles, f)

overall_time[f"pso_inverter_time_ood_patient_{ood_patient}"]['training_time'] = tok - tik
overall_history[f"pso_inverter_history_ood_patient_{ood_patient}"] = history_pso

# Defining the run: save time and history
stop = time.time()
overall_time['overall_time'] = stop - start
with open(os.path.join(general_reports_dir, f'timing.pkl'), 'wb') as f:
    pickle.dump(overall_time, f)
with open(os.path.join(general_reports_dir, f'overall_history.pkl'), 'wb') as f:
    pickle.dump(overall_history, f)

print(f"Total time: {stop - start}")
util_general.notification_ifttt(info=f"PSO Inverter Stopped, Time elapsed:{stop - start}")

print("May be the force with you!")
