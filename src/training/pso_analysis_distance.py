# -----------------------------------
# Author
#   Lorenzo Tronchin
# Script description
#
# Date
#   20/04/2022
# -----------------------------------

import sys
import pandas as pd

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import pickle
import time
import yaml
import os
import torch
import matplotlib.pyplot as plt

from src.utils import util_general
from inverter.utils_ae import util_inverter
from src.utils import util_latent_analysis
import math
from tqdm import tqdm
import seaborn as sns

import numpy as np

plt.rcParams.update({'font.size': 12})

# Configuration file
debug = 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
print("\nUpload configuration file")
if debug == 'develop':
    with open('./configs/dcgan_mnist.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    worker = cfg['device']['worker']
    device_type = cfg['device']['device_type']
    dataset_name = cfg['data']['dataset']
else:
    args = util_general.get_args_gan_discovery()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    device_type = args.device_type
    dataset_name = args.dataset

# SUBMIT RUN:
print("\nSubmit run")
# - Create log dir
# - Get new id_exp
# - Save the configuration file
# - Initialize Logger
# - Copy the code
run_module = os.path.basename(__file__)
if id_exp is None:
    run_id = util_general.get_next_run_id_local(os.path.join('log_run', dataset_name), run_module)  # GET run id
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
print(f'Configuration args: {id_exp, worker, device_type, dataset_name}')
print('Destination path: {}/{}'.format(dataset_name, run_name))

# Seed everything
print("\nSeed all")
util_general.seed_all(cfg['seed'])

# Parameters
print("\nParameters")
image_size = cfg['data']['image_size']
channel = cfg['data']['channel']
iid_class = cfg['data']['iid_classes']
ood_class = cfg['data']['ood_classes']

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

# Architecture
print("\nModel architecture")
print("Build encoder")
source_model_dir = util_general.define_source_path(cfg['data']['model_dir'], cfg['data']['dataset'], source_id_run=int('00002'), source_run_module='inverter.py')
E = util_inverter.Encoder(enc_dim=cfg['model_inverter']['latent_space'], channels_img=channel)
E.load_state_dict(torch.load(os.path.join(source_model_dir, f"encoder.pt"), map_location=torch.device(device)))
E.to(device)
# Generator
print("Build generator")
source_model_dir = util_general.define_source_path(cfg['data']['model_dir'], cfg['data']['dataset'], source_id_run=int('00001'), source_run_module='dcgan.py')
checkpoint_g = torch.load(os.path.join(source_model_dir, 'checkpoint_g.tar'),   map_location=torch.device(device))
G = util_inverter.Generator(z_dim=cfg['trainer_gan']['z_dim'], channels_img=channel, features_g=cfg['model_gan']['network']['units_gen'])
G.load_state_dict(checkpoint_g['model_state_dict'])
G.to(device)
E.eval()
G.eval()

print("\nStart analysis")
print("Upload iid data")
data = np.array([], dtype='float32')
label_iid = np.array([], dtype='uint8')
source_interim_dir = util_general.define_source_path(cfg['data']['interim_dir'], cfg['data']['dataset'],
                                                     source_id_run=int('00001'), source_run_module='pso_discovery.py')
for label in iid_class:  # per label
    # print(f"iid_class:{label}")
    data_iid = np.ones((cfg['trainer_pso']['n_particles'], cfg['trainer_pso']['dim_space']), dtype='float32')
    with open(os.path.join(source_interim_dir, f'particles_position_iic_class_{label}.pkl'), 'rb') as f:
        history_particles = pickle.load(f)
    for particle_idx, p_key in enumerate(history_particles.keys()):
        data_iid[particle_idx, :] = history_particles[p_key].iloc[-1, :]  # with -1 we select the last iteration
    if data.size == 0:
        data = data_iid.copy()
        label_iid = np.repeat(label, data_iid.shape[0])
    else:
        data = np.concatenate([data, data_iid], axis=0)
        label_iid = np.concatenate([label_iid, np.repeat(label, data_iid.shape[0])], axis=0)
data = pd.DataFrame(data)

print("\nUpload ood data")
#source_interim_dir = util_general.define_source_path(cfg['data']['interim_dir'], cfg['data']['dataset'],  source_id_run=int('00001'),  source_run_module='regularize_inverter.py')
source_interim_dir = util_general.define_source_path(cfg['data']['interim_dir'], cfg['data']['dataset'], source_id_run=int('00002'), source_run_module='ood_extractor.py')
try:
    with open(os.path.join(source_interim_dir, f'particles_position_ood.pkl'), 'rb') as f:
        data_ood = pickle.load(f)
except FileNotFoundError:
    for idx, label in enumerate(ood_class):  # per label
        with open(os.path.join(source_interim_dir, f'particles_position_ood_class_{str(label)}.pkl'), 'rb') as f:
            data_ood_class = pickle.load(f)
        data_ood_class = pd.concat([data_ood_class, pd.DataFrame(np.repeat(label, repeats=data_ood_class.shape[0], axis=0), columns=['label'])], axis=1)
        if idx == 0:
            data_ood = data_ood_class
        else:
            data_ood =  pd.concat((data_ood, data_ood_class), ignore_index=True)

print("Distance analysis")
fig1=plt.figure()
fig2=plt.figure()

# Mutual analysis
X1 = data_ood[data_ood['label'] == 4].drop(['label'], axis=1)
X2 = data_ood[data_ood['label'] == 5].drop(['label'], axis=1)
mse_mutual_distance = util_latent_analysis.mutual_distance(X1=X1[:250], X2=X2[:250])

idx = np.argsort(mse_mutual_distance, axis=0)
mse_sorted = mse_mutual_distance[idx, 0]
plt.figure(fig1.number)
plt.plot(mse_sorted, color='g')
plt.xlabel('pair index')
plt.ylabel('mse value')

plt.figure(fig2.number)
sns.distplot(mse_sorted, color='g')
plt.xlabel('mse value')
plt.ylabel('counts')

for label, c in zip(ood_class, ['r', 'b']):
    X =  data_ood[data_ood['label'] == label].drop(['label'], axis=1)
    X = X[:250]
    epoch = 0
    step = 0
    n = X.shape[0]
    k = 2
    total  = int(math.factorial(n)/(math.factorial(2) * math.factorial(n - k)))

    mse = np.empty((total, 1), dtype=np.dtype('float32'))
    for step_1, x1 in enumerate(X.to_numpy()):
        skip = np.arange(start=0, stop=step_1 + 1, step=1, dtype=np.dtype('int64'))
        with tqdm(total=n - skip.shape[0], desc=f'Latent vector passed {epoch + 1}/{n}, class_ood {label}', unit='latent vectors') as pbar:
            for step_2, x2 in enumerate(X.to_numpy()):  # Iterate over the batches of the dataset
                if step_2 in skip:
                    continue
                mse[step] = np.linalg.norm(x1 - x2)
                step  += 1
                pbar.update(1)
        epoch += 1

    idx = np.argsort(mse, axis=0)
    mse_sorted = mse[idx, 0]

    plt.figure(fig1.number)
    plt.plot(mse_sorted, color=c, label=label)
    plt.xlabel('pair index')
    plt.ylabel('mse value')

    plt.figure(fig2.number)
    sns.distplot(mse_sorted, color=c)
    plt.xlabel('mse value')
    plt.ylabel('counts')

plt.show()
fig1.savefig(os.path.join(general_reports_dir, f"paiwise_mse.png"), dpi=400, format='png')
fig2.savefig(os.path.join(general_reports_dir, f"latent_kde_distribution.png"), dpi=400, format='png')

# Defining the run: save time and history
stop = time.time()
overall_time['overall_time'] = stop - start
with open(os.path.join(general_reports_dir, 'timing.pkl'), 'wb') as f:
    pickle.dump(overall_time, f)
with open(os.path.join(general_reports_dir, 'overall_history.pkl'), 'wb') as f:
    pickle.dump(overall_history, f)
print(f"Total time: {stop - start}")
util_general.notification_ifttt(info=f"May be the force with you! Time elapsed:{stop - start}")
print("May be the force with you!")


