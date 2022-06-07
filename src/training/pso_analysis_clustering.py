# -----------------------------------
# Author
#   Lorenzo Tronchin
# Script description
#
# Date
#   20/04/2022
# -----------------------------------

#  Libraries
print('Import the library')
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import matplotlib.pyplot as plt
import os
import argparse
import yaml
import time
import pickle
import numpy as np
import pandas as pd

import torch

from src.utils import util_general
from src.utils import util_latent_analysis
from src.pso import util_pso_analysis


# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--dataset", help="Dataset to upload", type=str, default="mnist")

    parser.add_argument("--latent_dim", help="Dimension of the GAN latent space", type=int, default=100)
    parser.add_argument("--path_gan", help="Path to the pretrained GAN", type=str,  default="./models/mnist/00001--dcgan.py")

    parser.add_argument("--path_inverter", help="Path to the pretrained Inverter", type=str,  default="./models/mnist/00001--inverter.py")
    parser.add_argument("--path_iid_pso_discovery", help="Path to the PSO particles/latent positions of iid images", type=str, default="./data/interim/00001--pso_discovery.py")
    parser.add_argument("--ood_analysis", help="ood to perform out of distribution analysis, iid otherwise", type=str,   default="ood")

    parser.add_argument("--clustering_algorithm", help="Clustering algorithm for PSO analysis", type=str, default='em')

    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()

# Configuration file
print("Upload configuration file")
debug = 'develop' # 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
if debug == 'develop':
    with open('./configs/dcgan_mnist.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    worker = cfg['device']['worker']
    dataset_name = cfg['data']['dataset']

    latent_dim = cfg['trainer_gan']['z_dim']
    gan_dir = cfg['prerequisites']['model_gan']
    inverter_dir = cfg['prerequisites']['model_inverter']

    iid_pso_discovery_dir = cfg['prerequisites']['iid_pso_discovery']
    ood_analysis =  cfg['analysis_pso']['type']

    clustering_algorithm = cfg['trainer_pso_analysis']['clustering_algorithm']  # todo add folder to ood patient slice (path_ood_patient)
else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    dataset_name = args.dataset

    latent_dim = args.latent_dim
    gan_dir = args.path_gan
    inverter_dir = args.path_inverter

    iid_pso_discovery_dir = args.path_iid_pso_discovery
    ood_analysis = args.ood_analysis

    clustering_algorithm = args.clustering_algorithm

# Submit run:
print("Submit run")
new_root_dir_gan=gan_dir.split(sep='/')[-1] # NEW ROOT DIR!
new_root_dir_pso_discovery = iid_pso_discovery_dir.split('/')[-1]
new_root_dir = os.path.join(new_root_dir_gan, new_root_dir_pso_discovery)

run_module = os.path.basename(__file__)
iid_pso_discovery_dir = os.path.join(cfg['data']['interim_dir'], dataset_name, gan_dir.split(sep='/')[-1], iid_pso_discovery_dir.split('/')[-1])
ood_inverter_dir = os.path.join(cfg['data']['interim_dir'], dataset_name, gan_dir.split(sep='/')[-1],  inverter_dir.split(sep='/')[-1], f'00001--{ood_analysis}_extractor.py')

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
# Seed everything
print("\nSeed all")
util_general.seed_all(cfg['seed'])

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
print(f"gan pretrained dir: {gan_dir}")
print(f"iid pso particles dir: {iid_pso_discovery_dir}")
print(f"ood latent positions dir: {ood_inverter_dir}")
print(f"iid classes:{iid_class}")
print(f"ood classes:{ood_class}")

print(f"clustering algorithm: {clustering_algorithm}")

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


print("Start analysis")

data, label_iid = util_pso_analysis.upload_pso_particles(path_dir=iid_pso_discovery_dir, n_particles=cfg['trainer_pso']['n_particles'], latent_dim=latent_dim, patient_classes=iid_class)

model= util_pso_analysis.get_clustering_algorithm(model_name=clustering_algorithm, data=data, data_iid_label=label_iid, n_components = data.shape[1], dim_red_algorithm='no_transformation')
with open(os.path.join(model_dir, f'{clustering_algorithm}.pkl'), 'wb') as f:
    pickle.dump(model, f)
if latent_dim==2:
    util_latent_analysis.plot_latent_space(plot_training_dir=plot_training_dir, data_iid=data.to_numpy(),  data_iid_label=label_iid)
    if clustering_algorithm == 'em':
        util_latent_analysis.plot_ellipsoids(plot_training_dir=plot_training_dir, X=data, Y_=model.predict(data), means=model.means_, covariances=model.covariances_, dim_red_algorithm="Gaussian Mixture")

for dim_red_algorithm in ['pca', 'umap']:
    print(dim_red_algorithm)

    print("iid analysis")
    model_dim_red, reducer, reduced_data_iid = util_pso_analysis.get_clustering_algorithm(model_name=clustering_algorithm, data=data, data_iid_label=label_iid,  n_components=2, dim_red_algorithm=dim_red_algorithm)
    util_latent_analysis.plot_latent_space(plot_training_dir=plot_training_dir, data_iid=reduced_data_iid, data_iid_label=label_iid, dim_reduction_algorithm=dim_red_algorithm)
    if clustering_algorithm == 'em':
        util_latent_analysis.plot_ellipsoids(plot_training_dir, reduced_data_iid, model_dim_red.predict(reduced_data_iid), model_dim_red.means_, model_dim_red.covariances_, dim_red_algorithm)

    if ood_analysis == 'ood':
        print(f"ood analysis")
        for label in ood_class: # per label
            label_ood = str(label)
            print(f"ood_class:{label_ood}")
            try:
                with open(os.path.join(ood_inverter_dir, f'particles_position_ood.pkl'), 'rb') as f:
                    data_ood = pickle.load(f)
                data_ood = data_ood[data_ood.iloc[:, -1] == label]
                data_ood = data_ood.drop(data_ood.columns[-1], axis=1)
            except FileNotFoundError:
                with open(os.path.join(ood_inverter_dir, f'particles_position_ood_class_{label_ood}.pkl'), 'rb') as f:
                    data_ood = pickle.load(f)

            if latent_dim == 2:
                util_latent_analysis.plot_latent_space(plot_training_dir=plot_training_dir, data_iid=data.to_numpy(),  data_iid_label=label_iid, data_ood=data_ood.to_numpy(), data_ood_label=label_ood)

            reduced_data_ood = reducer.transform(data_ood)
            util_latent_analysis.plot_latent_space(plot_training_dir=plot_training_dir, data_iid=reduced_data_iid, data_iid_label=label_iid, data_ood=reduced_data_ood,  data_ood_label=label_ood, dim_reduction_algorithm=dim_red_algorithm)
    elif ood_analysis == 'iid':
        print(f"iid analysis")
        for label in iid_class:  # per label
            label_iid_inverted = str(label)
            print(f"iid_class:{label_iid_inverted}")
            with open(os.path.join(ood_inverter_dir, f'particles_position_iid_class_{label_iid_inverted}.pkl'), 'rb') as f:
                data_iid_inverted = pickle.load(f)

            if latent_dim == 2:
                util_latent_analysis.plot_latent_space(plot_training_dir=plot_training_dir, data_iid=data.to_numpy(),  data_iid_label=label_iid, data_ood=data_iid_inverted.to_numpy(), data_ood_label=label_iid_inverted)

            reduced_data_iid_inverted = reducer.transform(data_iid_inverted)
            util_latent_analysis.plot_latent_space(plot_training_dir=plot_training_dir, data_iid=reduced_data_iid,  data_iid_label=label_iid, data_ood=reduced_data_iid_inverted,   data_ood_label=label_iid_inverted,  dim_reduction_algorithm=dim_red_algorithm)
    else:
        raise ValueError(ood_analysis)

# Defining the run: save time and history
print("May be the force with you!")