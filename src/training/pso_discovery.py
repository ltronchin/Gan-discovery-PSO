# -----------------------------------
# Author
#   Lorenzo Tronchin
# Script description
#   Discover concept in GAN latent space throught PSO
# Date
#   13/04/2022
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

import numpy as np

import torch

from src.pso import util_cnn
from src.utils import util_general
from src.utils import util_dcgan
from src.pso import util_discovery
from src.pso import util_pso
from src.utils import util_report

def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--dataset", help="Dataset to upload", type=str, default="mnist")

    parser.add_argument("--latent_dim", help="Dimension of the GAN latent space", type=int, default=100)
    parser.add_argument("--path_cnn", help="Path to the pretrained CNN/CNNs", type=str,  default= "./models/mnist/00001--cnn_multipatient.py")
    parser.add_argument("--path_gan", help="Path to the pretrained GAN", type=str, default="./models/mnist/00001--dcgan.py")

    parser.add_argument("--w_ine", help="Inertia coefficient of particle", type=float, default= 0.73)
    parser.add_argument("--w_cogn", help="Cognitive coefficient of particle", type=float, default= 1.496)
    parser.add_argument("--w_soci", help="Social coefficient of particle", type=float, default= 1.496)
    parser.add_argument("--schedule_ine", help="Activate the schedule over iterations of inertia", type=bool, default=False)

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

    w_ine = cfg['trainer_pso']['w_inertia']
    w_cogn = cfg['trainer_pso']['w_cognitive'],
    w_soci = cfg['trainer_pso']['w_social']
    schedule_ine = cfg['trainer_pso']['schedule_inertia']
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

    w_ine = args.w_ine
    w_cogn =  args.w_cogn
    w_soci =  args.w_soci
    schedule_ine = args.schedule_ine

# Submit run:
print("Submit run")
new_root_dir=gan_dir.split(sep='/')[-1]
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

n_particles = cfg['trainer_pso']['n_particles']

# Useful print
print(f"id_exp: {id_exp}")
print(f"Latent_dim: {latent_dim}")
print(f"cnn_pretrained_dir: {cnn_dir}")
print(f"gan_pretrained_dir: {gan_dir}")
print(f"iid classes:{iid_class}")
print(f"ood classes:{ood_class}")
print(f"Number of particles:{n_particles}")
print(f"Inertia: {w_ine}")
print(f"Cognitive: {w_cogn}")
print(f"Social:{w_soci}")
print(f"Schedule inertia:{schedule_ine}")

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

# Start Training
print("Start training")
util_general.notification_ifttt(info=f"GAN discovery started")
for label in iid_class:
    print(f"iid_class:{label}")
    # Dir
    general_reports_label_dir = os.path.join(general_reports_dir, str(label))
    util_general.create_dir(general_reports_label_dir)
    plot_training_label_dir = os.path.join(plot_training_dir, str(label))
    util_general.create_dir(plot_training_label_dir)

    # CNN
    print("Upload cnn")
    if 'cnn_multipatient' in cnn_dir.split(sep='/')[-1]:
        print('Multipatient cnn')
        model_cnn = util_cnn.get_cnn(model_name=cfg['model_cnn']['model_name'], image_channels=channel,  iid_classes=iid_class, n_class=len(iid_class), img_dim=image_size, cnn_args=cfg['model_cnn']['network'])
        model_cnn.load_state_dict(torch.load(os.path.join(cnn_dir, f"model.pt"), map_location=torch.device(device)))
    else:
        model_cnn = util_cnn.get_cnn(model_name=cfg['model_cnn']['model_name'], image_channels=channel,  iid_classes=iid_class, n_class=2, img_dim=image_size, cnn_args=cfg['model_cnn']['network'])
        model_cnn.load_state_dict(torch.load(os.path.join(cnn_dir, f"model_{label}.pt"), map_location=torch.device(device)))
    model_cnn = model_cnn.to(device)

    # GAN
    print("Upload gan generator")
    checkpoint_g = torch.load(os.path.join(gan_dir, 'best_g.tar'), map_location=torch.device(device))
    generator = util_dcgan.Generator(z_dim=latent_dim, channels_img=channel, features_g=cfg['model_gan']['network']['units_gen'])
    generator.load_state_dict(checkpoint_g['model_state_dict'])
    generator = generator.to(device)

    # PSO
    print("Start PSO")
    discovery = util_discovery.Discovery(iid_class=label, model_gan=generator, model_cnn=model_cnn, device=device, iid_classes = iid_class, obj_fun_threshold=0)

    swarm = util_pso.Swarm(
        plot_training_dir = plot_training_label_dir,
        discovery=discovery,
        num_particles=n_particles,
        n_iterations=cfg['trainer_pso']['n_iterations'],
        dim_space=latent_dim,
        device=device,
        tolerance = cfg['trainer_pso']['tolerance'],
        w_inertia = w_ine,
        w_cogn = w_cogn,
        w_soci = w_soci
    )

    tik = time.time()
    history, history_particles, history_particles_vel, last_iteration = swarm.optimize(schedule_inertia=schedule_ine, early_stopping=cfg['trainer_pso']['early_stopping'])
    tok = time.time()

    # Plot Discovery
    print("Plot training results")
    util_report.plot_pso_convergence(plot_training_label_dir=general_reports_label_dir,  global_best_val=swarm.g_best_val_dummy)
    util_report.plot_training(history=history, plot_training_dir=general_reports_label_dir)
    print("Plot latent space")
    if latent_dim == 2:
        fitness_grid, img_grid = util_report.plot2d(plot_training_dir=plot_training_label_dir, history_particles=history_particles, fitness=discovery.fitness, g_best_pos=swarm.g_best_pos, grid_range=5, color='#000')
        with open(os.path.join(general_reports_label_dir, f'fitness_grid.pkl'), 'wb') as f:
            pickle.dump(fitness_grid, f)
        with open(os.path.join(general_reports_label_dir, f'img_grid.pkl'), 'wb') as f:
            pickle.dump(img_grid, f)
        util_report.make_gif_from_folder(plot_training_dir=plot_training_label_dir,  img_to_upload_dir=plot_training_label_dir, filename_result='2dspace_latent.gif', filename_source='2d_plot')

    util_report.plot_feature(plot_training_dir=plot_training_label_dir, history_particles=history_particles, dim_space=latent_dim, iteration=last_iteration)
    util_report.plot_features_last_iteration(plot_training_dir=plot_training_label_dir, history_particles=history_particles, dim_space=latent_dim)
    print("Make gif")
    util_report.make_gif_from_folder(plot_training_dir=plot_training_label_dir, img_to_upload_dir=plot_training_label_dir)

    with open(os.path.join(interim_dir, f'particles_position_iid_class_{label}.pkl'), 'wb') as f:
        pickle.dump(history_particles, f)

    overall_time[f"class_{label}"]['training_time'] = tok - tik
    overall_history[f"class_{label}"] = history

# Defining the run: save time and history
stop = time.time()
overall_time['overall_time'] = stop - start
with open(os.path.join(general_reports_dir, f'timing.pkl'), 'wb') as f:
    pickle.dump(overall_time, f)
with open(os.path.join(general_reports_dir, f'overall_history.pkl'), 'wb') as f:
    pickle.dump(overall_history, f)

print(f"Total time: {stop - start}")
util_general.notification_ifttt(info=f"GAN Discovery Stopped, Time elapsed:{stop - start}")

print("May be the force with you!")
