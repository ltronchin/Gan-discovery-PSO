"""
Followed the implementation at the following link:
https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=rMl2DjQs7hxT
https://github.com/ritheshkumar95/pytorch-vqvae
"""
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import argparse
import yaml
import time
import numpy as np
import os
import pickle
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from src.utils import util_general
from src.utils import util_data

from src.inverter.utils_vq_vae import util_training
from src.inverter.utils_vq_vae import util_model
from src.inverter.utils_vq_vae import util_report

class pso_particles:
    def __init__(self, cfg, iid_class):
        self.cfg = cfg
        self.iid_class = iid_class

    def upload_pso_disentangled_space(self):
        print("Upload iid data from pso discovery")
        data = np.array([], dtype='float32')
        label_iid = np.array([], dtype='uint8')
        source_interim_dir = util_general.define_source_path(self.cfg['data']['interim_dir'],
                                                             self.cfg['data']['dataset'], source_id_run=int('00001'),
                                                             source_run_module='pso_discovery.py')
        for label in self.iid_class:  # per label
            print(f"iid_class:{label}")
            data_iid = np.ones((self.cfg['trainer_pso']['n_particles'], self.cfg['trainer_pso']['dim_space']),  dtype='float32')
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
        return data


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
    with open('./configs/vqvae.yaml') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = cfg['id_exp']
    device_type = cfg['device']['device_type']
    worker = cfg['device']['worker']
    dataset_name = cfg['data']['dataset']
else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    device_type = args.device_type # todo notused!
    worker = args.gpu
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

# Seed everything
print("\nSeed all")
util_general.seed_all(cfg['seed'])

# Parameters
print("\nParameters")
image_size = cfg['data']['image_size']
channel = cfg['data']['channel']
iid_class = cfg['data']['iid_classes']
ood_class  = cfg['data']['ood_classes']

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
util_general.create_dir(os.path.join(general_reports_dir, 'logs')) # create logs folder

writer = SummaryWriter(os.path.join(general_reports_dir, 'logs/'))

# Data loaders>upload images from ood distribution
print("\nCreate dataloader")
iid_train_dataset, iid_val_dataset = util_data.get_public_dataset(dataset_name=dataset_name, data_dir=data_dir, drange_net=cfg['data']['drange_net'], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=iid_class)
ood_train_dataset, ood_val_dataset = util_data.get_public_dataset(dataset_name=dataset_name, data_dir=data_dir, drange_net=cfg['data']['drange_net'], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=ood_class)
# Define the dataloader
train_loader = DataLoader(dataset=iid_train_dataset, batch_size=cfg['trainer']['batch_size'], shuffle=True, drop_last=True)
valid_iid_loader = DataLoader(dataset=iid_val_dataset, batch_size=cfg['trainer']['batch_size'], shuffle=False,  drop_last=True)
valid_ood_loader = DataLoader(dataset=ood_train_dataset, batch_size=cfg['trainer']['batch_size'], shuffle=False,  drop_last=True)
data_loaders = {
    'train': train_loader,
    'val_iid': valid_iid_loader,
    'val_ood': valid_ood_loader
}
print("Sanity check on data")
for dset, loader in zip(['Training', 'Validation'], [train_loader, valid_iid_loader]):
    images, labels = iter(loader).next()
    print(f"{dset} Min: {torch.min(images)}, Max: {torch.max(images)}")
assert torch.min(images) == cfg['data']['drange_net'][0]
assert torch.max(images) == cfg['data']['drange_net'][1]

# Upload pso particles
pso_p = pso_particles(cfg=cfg, iid_class = iid_class)
data_pso = pso_p.upload_pso_disentangled_space()

# Architecture
print("\nModel architecture")
num_channels = cfg['data']['channel']

model = util_model.get_model(name = 'vqvae_dcgan',
                     channels_img=num_channels,
                     embedded_dim=cfg['model']['latent_space']['embedding_dim'],
                     num_embedding=cfg['model']['latent_space']['num_embedding'],
                     data_pso=data_pso)

model = model.to(device)

if cfg['model']['train_inverter'] is not None:
    print("Upload generator weights")
    checkpoint_g = torch.load(os.path.join(cfg['pretrained_input']['model_gan'], 'checkpoint_g.tar'), map_location=torch.device(device))
    generator = util_model.Generator(embedded_dim=cfg['model']['latent_space']['embedding_dim'], channels_img=channel,  features_g=cfg['model_gan']['network']['units_gen'])
    generator.load_state_dict(checkpoint_g['model_state_dict'])
    generator.to(device)
    model.decoder = util_model.freeze_parameters(model=generator)


optimizer = util_model.get_opti(model.parameters(), **cfg['trainer']['optimizer'])

print("\nStart training")
tik = time.time()

# Fixed images for Tensorboard
fixed_images, _ = next(iter(train_loader))

fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
writer.add_image('Original', fixed_grid, 0)

reconstruction = util_training.generate_samples(images=fixed_images, model=model, device=device) # generate the samples
grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
writer.add_image('Reconstruction', grid, 0)

history = {'train_loss_recons': [], 'train_loss_vq': [], 'val_ood_loss_recons': [], 'val_ood_loss_vq': []}
fixed_noise = torch.randn((32, 100, 1, 1)).to(device)
best_loss = -1.
for epoch in range(cfg['trainer']['epochs']):
    print(f'Epoch:{epoch + 1}')
    _, train_loss_recons, train_loss_vq = util_training.train(data_loader=train_loader, model=model, optimizer=optimizer, beta=cfg['trainer']['beta'], device=device, train_inverter=cfg['model']['train_inverter'])
    writer.add_scalar('loss/train/reconstruction', train_loss_recons, epoch + 1) # Logs
    writer.add_scalar('loss/train/quantization', train_loss_vq, epoch + 1)
    util_report.show_images(general_reports_dir=general_reports_dir, epoch=epoch, data_loaders=train_loader, model=model, device=device, phase='train', n_img=10)

    valid_loss_recons, valid_loss_vq = util_training.valid(data_loader=valid_ood_loader, model=model, device=device)
    writer.add_scalar('loss/valid/reconstruction', valid_loss_recons, epoch) # Logs
    writer.add_scalar('loss/valid/quantization', valid_loss_vq, epoch)
    util_report.show_images(general_reports_dir=general_reports_dir, epoch=epoch, data_loaders=valid_ood_loader,  model=model, device=device, phase='val_ood', n_img=10)
    util_report.show_images(general_reports_dir=general_reports_dir, epoch=epoch, data_loaders=valid_iid_loader,  model=model, device=device, phase='val_iid', n_img=10)

    history['train_loss_recons'].append(train_loss_recons)
    history['train_loss_vq'].append(train_loss_vq)
    history['val_ood_loss_recons'].append(valid_loss_recons)
    history['val_ood_loss_vq'].append(valid_loss_vq)
    util_report.plot_training(history=history, plot_training_dir=plot_training_dir)
    util_report.show_gan_images(general_reports_dir=general_reports_dir, epoch=epoch, noise=fixed_noise, decoder=model.decoder)

    reconstruction = util_training.generate_samples(images=fixed_images, model=model, device=device)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('Reconstruction', grid, epoch + 1)

    if (epoch == 0) or (valid_loss_recons < best_loss):
        best_loss = valid_loss_recons
        with open('{0}/best_{1}.pt'.format(model_dir, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)
    with open('{0}/model_{1}.pt'.format(model_dir, epoch + 1), 'wb') as f:
        torch.save(model.state_dict(), f)

tok = time.time()

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
