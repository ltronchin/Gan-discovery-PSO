# -----------------------------------
# Author
#   Lorenzo Tronchin
# Script description
#
# Date
#   13/04/2022
# -----------------------------------

import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

# Welcome
from datetime import datetime
now = datetime.now()
date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
print("Hello!",date_time)

import torch

from src.utils import util_report
from src.utils import util_general
from src.utils import util_latent_analysis
import numpy as np

import pickle
import time
import yaml
import os
import pandas as pd

# Configuration file
debug = 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
print("Upload configuration file")
if debug == 'develop':
    with open('./configs/dcgan_mnist.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    worker = cfg['device']['worker']
    device_type = cfg['device']['device_type']
    dataset_name = cfg['data']['dataset']
    model_name = cfg['model_pso_analysis']['model_name']
else:
    args = util_general.get_args_gan_discovery()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    device_type = args.device_type
    dataset_name = args.dataset
    model_name = args.model

# Seed everything
print("Seed all")
util_general.seed_all(cfg['seed'])

# Parameters
image_size = cfg['data']['image_size']
channel = cfg['data']['channel']
iid_class = cfg['data']['iid_classes']

# Register and history
overall_time = util_general.nested_dict()
overall_history = {}
start = time.time()

# Device
device = torch.device(worker if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(f'device: {device}')

# Files and Directories
data_dir = os.path.join(cfg['data']['data_dir'], dataset_name)
interim_dir = os.path.join(cfg['data']['interim_dir'], dataset_name, id_exp)
util_general.create_dir(interim_dir)
model_dir = os.path.join(cfg['data']['model_dir'], dataset_name, id_exp)
util_general.create_dir(model_dir)
reports_dir = os.path.join(cfg['reports']['reports_dir'], dataset_name, model_name, id_exp)
util_general.create_dir(reports_dir)
plot_training_dir = os.path.join(reports_dir, "training_plot")
util_general.create_dir(plot_training_dir)
general_reports_dir = os.path.join(reports_dir, "general")
util_general.create_dir(general_reports_dir)

# Save the configuration file
with open('configuration.yaml', 'wb') as f:
    pickle.dump(cfg, f)

util_general.notification_ifttt(info=f"PSO analysis started")
for i in range(cfg['trainer_pso']['n_iterations']): # per iteration
    print(f"pso_iteration:{i}")

    data = np.array([], dtype='float32')
    label_iid = np.array([], dtype='uint8')

    for label in iid_class: # per label
        print(f"iid_class:{label}")
        data_iid = np.ones((cfg['trainer_pso']['n_particles'], cfg['trainer_pso']['dim_space']), dtype='float32')
        with open(os.path.join(interim_dir, f'particles_position_iic_class_{label}.pkl'), 'rb') as f:
            history_particles = pickle.load(f)

        if i < len(history_particles['particle_0']):
            for particle_idx, p_key in enumerate(history_particles.keys()):
                data_iid[particle_idx, :] = history_particles[p_key].iloc[i, :]
        else:
            print(f"IndexError! Iteration not available for the current iid_class {label}, let's select the last one")
            for particle_idx, p_key in enumerate(history_particles.keys()):
                data_iid[particle_idx, :] = history_particles[p_key].iloc[-1, :]

        if data.size == 0:
            data = data_iid.copy()
            label_iid = np.repeat(label, data_iid.shape[0])
        else:
            data = np.concatenate([data, data_iid], axis=0)
            label_iid = np.concatenate([label_iid, np.repeat(label, data_iid.shape[0])], axis=0)

    data = pd.DataFrame(data)

    util_latent_analysis.pca_fun(plot_training_dir=plot_training_dir, data=data, data_iid_label=label_iid, iteration=i)
    util_latent_analysis.umap_fun(plot_training_dir=plot_training_dir, data=data, data_iid_label=label_iid, iteration=i)

util_report.make_gif_from_folder(plot_training_dir=plot_training_dir, img_to_upload_dir=plot_training_dir,filename_result='pca_space.gif', filename_source='pca_space' )

# Defining the run: save time and history
stop = time.time()
overall_time['overall_time'] = stop - start
with open(os.path.join(general_reports_dir, f'timing_{model_name}.pkl'), 'wb') as f:
    pickle.dump(overall_time, f)
with open(os.path.join(general_reports_dir, f'overall_history_{model_name}.pkl'), 'wb') as f:
    pickle.dump(overall_history, f)

print(f"Total time: {stop - start}")
util_general.notification_ifttt(info=f"May be the force with you! Time elapsed:{stop - start}")

print("May be the force with you!")

