# Author: Lorenzo Tronchin
# Data: Claro Retrospettivo 512x512
# Script to prepare medical data for StyleGAN

# Input:
# Output: folder claro_retrospettivo_tif

import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.utils import util_general
from src.utils import util_medical_data
from src.utils import logger

import time
import yaml
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration file
debug = 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")
print("Upload configuration file")
if debug == 'develop':
    with open('./configs/claro_preprocess.yaml') as file:
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

# Seed everything
print("Seed all")
util_general.seed_all(cfg['seed'])

# Parameters


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
interim_dir = os.path.join(cfg['data']['interim_dir'], dataset_name)
util_general.create_dir(interim_dir)
interim_dir_stylegan =  os.path.join(interim_dir, 'stylegan')
util_general.create_dir(interim_dir_stylegan)

#Logger
logger = logger.setup_logger(interim_dir, 'data_preparation.log', 'data_preparation_logger')

# Data loaders
logger.info(f'Dataset: {dataset_name}')
data_raw = pd.read_excel(os.path.join(interim_dir, f'patients_info_{dataset_name}.xlsx'), index_col=0)
id_patients_slice = pd.Series([row.split(os.path.sep)[1].split('.tif')[0] for row in data_raw['image']])
id_patients = pd.Series([idp.split('_')[0] for idp in id_patients_slice.iloc])

box_data = pd.read_excel(cfg["data"]["box_file"])
id_patients_slice_box = box_data['img ID']

id_patients_slice_lung = pd.Series(np.intersect1d(id_patients_slice, id_patients_slice_box))

logger.info(f'Number of images: {len(id_patients_slice_lung)}')
logger.info(f'Number of patients: {len(np.unique(id_patients))}')
logger.info('Create dataloader')
dataset = util_medical_data.ImgDatasetPreparation(data=id_patients_slice_lung, cfg_data=cfg['data'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

with tqdm(total=len(dataloader.dataset),  unit='img') as pbar:
    for x, idp, ids in dataloader:  # Iterate over the batches of the dataset
        img = x.detach().cpu().numpy()[0][0]
        idp = idp[0]
        ids = ids[0]

        image = Image.fromarray(img)
        filename = idp + '_' + ids
        image.save(os.path.join(interim_dir_stylegan, f'{filename}.tif' ), 'TIFF' )

        pbar.update(x.shape[0])

logger.info("May be the force with you!")