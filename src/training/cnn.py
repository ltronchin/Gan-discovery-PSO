# -----------------------------------
# Author
#   Lorenzo Tronchin
# Script description
#   Training of shallow cnn
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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchsummary import summary

from src.pso import util_cnn
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
print("Parameters")
image_size = cfg['data']['image_size']
channel = cfg['data']['channel']
iid_class = cfg['data']['iid_classes']
ood_class  = cfg['data']['ood_classes']

# Useful print
print(f"id_exp: {id_exp}")
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
train_dataset, val_dataset = util_data.get_public_dataset(dataset_name=dataset_name, data_dir=data_dir, drange_net=[0, 1], general_reports_dir=general_reports_dir, image_size=image_size, channel=channel, iid_class=iid_class)
train_loader = DataLoader(train_dataset, batch_size=cfg["trainer_cnn"]["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size = cfg['trainer_cnn']['batch_size'], shuffle=True, drop_last=True)
data_loaders = {
        'train': train_loader,
        'val':val_loader
}

# Check on the data
print("Sanity check on data")
for dset, loader in zip(['Training', 'Validation'], [train_loader, val_loader]):
    images, labels = iter(loader).next()
    print(f"{dset} Min: {torch.min(images)}, Max: {torch.max(images)}")

# Upload requirements for the run (pretrained models, data, ...)

print('Training phase')
try:
    models = {}
    for label in iid_class:
        print(f'Try to upload cnn of iid_class: {label}')
        model = util_cnn.get_cnn(model_name=cfg['model_cnn']['model_name'], image_channels=channel, iid_classes=[0, 1], n_class=2, img_dim=image_size, cnn_args=cfg['model_cnn']['network'])
        model.load_state_dict(torch.load(os.path.join(model_dir, f"model_{label}.pt"), map_location=torch.device(device)))
        model = model.to(device)
        models[f"class_{label}"] = model
except FileNotFoundError:
    models = {}
    for label in iid_class:
        print(f'Training from scratch cnn of iid_class: {label}')
        # Architecture
        print("Model architecture")
        model = util_cnn.get_cnn(model_name=cfg['model_cnn']['model_name'], image_channels=cfg['data']['channel'], iid_classes=[0, 1], n_class=2,  img_dim=cfg['data']['image_size'], cnn_args=cfg['model_cnn']['network'])
        model=model.to(device)
        util_cnn.initialize_weights(model, cfg['model_cnn']['network']['cnn_initializer'])
        # if label == 0:
            # print(summary(model, (cfg["data"]["channel"], cfg["data"]["image_size"], cfg["data"]["image_size"]), device=device))

        # Optimizer and loss
        print("Optimizer and loss")
        # Loss
        criterion = nn.CrossEntropyLoss().to(device)
        # Optimizer
        optimizer = util_cnn.get_opti(model.parameters(), **cfg['trainer_cnn']['optimizer'])
        # LR Scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg['trainer_cnn']['scheduler']['mode'],  patience=cfg['trainer_cnn']['scheduler']['patience'])

        # Train model
        print(f"Start training class_label: {label}")
        util_general.notification_ifttt(info=f"cnn, class_{label} training started")
        tik = time.time()
        model, history = util_cnn.train_model(
            model=model, data_loaders=data_loaders, criterion=criterion,  optimizer=optimizer, scheduler=scheduler, num_epochs=cfg['trainer_cnn']['epochs'],
            early_stopping=cfg['trainer_cnn']['early_stopping'], model_dir=model_dir, device=device, label=label
        )
        tok = time.time()
        print(f"Training time: {tok-tik}")
        overall_time[f"class_{label}"]['training_time'] = tok - tik
        overall_history[f"class_{label}"] = history
        models[f"class_{label}"] = model
        # Plot Training
        util_report.plot_training(history=history, plot_training_dir=plot_training_dir, label=label)

    # Defining the run: save time and history
    stop = time.time()
    overall_time['overall_time'] = stop - start
    with open(os.path.join(general_reports_dir, f'timing.pkl'), 'wb') as f:
        pickle.dump(overall_time, f)
    with open(os.path.join(general_reports_dir, f'overall_history.pkl'), 'wb') as f:
        pickle.dump(overall_history, f)

    print(f"Total time: {stop - start}")
    util_general.notification_ifttt(info=f"May be the force with you! Time elapsed:{stop - start}")

# Evaluation phase
history = {}
print("\nEvaluation phase")
fig = plt.figure()
for label in  iid_class:
    print(f"Class label: {label}")
    running_correct = [] # per classifier

    for iid_class_classifier in iid_class: # pass test set through each classifier
        print(f"cnn: {iid_class_classifier}")
        running_batch_correct = []

        cnn = models[f"class_{iid_class_classifier}"]
        cnn.eval()
        cnn = cnn.to(device)
        for x_test, y_test in val_loader:  # Iterate over the batches of the dataset
            y_test = y_test == label  # binarize dataset>1 for class label of interest, 0 otherwise
            y_test = y_test.to(torch.uint8)

            x_test = x_test[y_test == 1]
            x_test = x_test.to(device)

            #y_test = y_test[y_test == 1].to(device)  # select only positive label
            with torch.no_grad():
                output = cnn(x_test.float())
                _, preds = torch.max(output, 1)
            running_batch_correct.append(sum(preds.detach().cpu().numpy()))
        running_correct.append(sum(running_batch_correct))

    plt.plot(running_correct, label=label)

plt.legend()
plt.xticks(np.arange(len(iid_class)), iid_class)
plt.xlabel("Classifiers")
plt.ylabel("Classifier activation per test set")
fig.savefig(os.path.join(general_reports_dir, "classifier_battery_tree.png"), dpi=400, format='png')
plt.show()
print("May be the force with you!")




