# -----------------------------------
# Author
#   Lorenzo Tronchin
# Script description
#   Training of the classifiers battery using the training embedding extracted from Convolutional Autoencoder
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.utils import util_general
from src.utils import util_data
from src.evaluation import util_cae

# Argument function
def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("-i", "--id_exp", help="Id of experiment to load models and datas", type=int, default=1)
    parser.add_argument("-g", "--gpu", help="CUDA device", type=str, default="cuda:0")
    parser.add_argument("--dataset", help="Dataset to upload", type=str, default="mnist")

    parser.add_argument("--path_latent_den_cae", help="Path to the embedding space of training and validation set extracted with CAE", type=str,  default="./data/interim/mnist/00001--cae.py")

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

    den_cae_dir = cfg['prerequisites']['latent_den_cae']
else:
    args = get_args()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    id_exp = args.id_exp
    worker = args.gpu
    dataset_name = args.dataset

    den_cae_dir = args.path_latent_den_cae

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
try:
    print("Upload embedding vector")
    h_tr = pd.read_csv(os.path.join(den_cae_dir, "encoded_samples_train.csv"), index_col=0)
    # Extract validation data
    h_te = pd.read_csv(os.path.join(den_cae_dir, "encoded_samples_valid.csv"), index_col=0)
except FileNotFoundError:
    # Data loaders
    print("Create dataloader")
    train_dataset, val_dataset = util_data.get_public_dataset(dataset_name=dataset_name, data_dir=data_dir, drange_net=[0, 1],  general_reports_dir=general_reports_dir, image_size=image_size, channel=channel,  iid_class=iid_class)
    train_loader = DataLoader(train_dataset, batch_size=cfg["trainer_ae"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=cfg['trainer_ae']['batch_size'], shuffle=True, drop_last=True)

    # Upload pretrained models: classifiers and encoder
    print("Upload Encoder and Decoder for evaluation phase")
    encoder, decoder = util_cae.load_autoencoder(model_dir=model_dir, latent_space=cfg['model_ae']['latent_space'], device=device)

    # Create encoded sample for training set
    h_tr = util_cae.create_encoded_sample(encoder=encoder, data=train_dataset, device=device)
    h_tr.to_csv(os.path.join(interim_dir, "encoded_samples_train.csv"))  # save the train embeddings
    util_cae.plot_feature_latent_space(general_reports_dir=general_reports_dir, encoded_samples=h_tr, dataset='Training')

    # Create encoded sample for validation set
    h_te = util_cae.create_encoded_sample(encoder=encoder, data=val_dataset, device=device)
    h_te.to_csv(os.path.join(interim_dir, "encoded_samples_valid.csv"))  # save the validation embeddings
    util_cae.plot_feature_latent_space(general_reports_dir=general_reports_dir, encoded_samples=h_te, dataset='Validation')

# Start Training
print("Start training")
val_size = int(len(h_tr)* 0.2) # let reserve 20% of our data for validation
thresholds = np.linspace(0.0, 0.9, num=90)
t_bin = 0.5
label_tr = np.unique(h_tr["label"])
clf_pts = {}
for label in label_tr:
    print(f'iid_class: {label}')
    # Create N new dataset with boolean label where N is equal to the number of class
    x = h_tr.iloc[:, :-1].to_numpy()
    y = np.array(h_tr["label"] == label, dtype="uint8")
    # Extract train data
    x_train = x[:-val_size]
    y_train = y[:-val_size]
    # Extract validation data
    x_valid = x[-val_size:]
    y_valid = y[-val_size:]

    clf =  KNeighborsClassifier(n_neighbors=5) #clf = SVC(gamma='auto', probability=True) #clf =  KNeighborsClassifier(n_neighbors=3)  #clf = tree.DecisionTreeClassifier()
    clf_pts[label] = clf.fit(x_train, y_train)

    # Error-reject curve
    p_rej_list = []
    p_error_list = []
    pred_bin =  clf.predict(x_valid)
    pred = clf.predict_proba(x_valid)[:, 1]
    reliability = np.abs(1 - (pred / t_bin))
    for ths in thresholds:
        # compute reliability
        mask = reliability > ths

        # compute the % of rejected
        p_rej =  (np.abs(np.sum(mask) - val_size) / val_size) * 100
        p_rej_list.append(p_rej)

        # compute the % of error
        p_error = (1 - accuracy_score(y_valid[mask], pred_bin[mask])) * 100
        p_error_list.append(p_error)

    plt.plot(p_rej_list, p_error_list, marker='o')
    #for i, txt in enumerate([x for x in thresholds]): # plot threshold over the curve
    #    plt.annotate(str(txt), (p_rej_list[i], p_error_list[i]))
    plt.title(label)
    plt.ylabel("% error")
    plt.xlabel("% rejection")
    plt.ylim([0, 30])
    plt.savefig(os.path.join(general_reports_dir, f"error_reject_curve_{label}.png"), dpi=400, format='png')
    plt.show()

# Save to disk
with open(os.path.join(model_dir,"classifiers"), "wb") as f:
    pickle.dump(clf_pts, f)

# Evaluation phase
label_te = np.unique(h_te["label"])
h_te_pts = {label: h_te[h_te["label"] == label] for label in label_te}
history = {}
print("\nEvaluation phase")
for label in label_te:
    print(f'iid_class: {label}')
    running_correct = []
    x_test = h_te_pts[label].iloc[:, :-1].to_numpy()

    for classifier in label_te: # pass test set through each classifier
        running_correct.append(sum(clf_pts[classifier].predict(x_test)))

    history[label] = running_correct
    plt.plot(history[label], label=label)

plt.legend()
plt.xticks(np.arange(len(iid_class)), iid_class)
plt.xlabel("Classifiers")
plt.ylabel("Classifier activation per test set")
plt.savefig(os.path.join(general_reports_dir, "classifier_battery_tree.png"), dpi=400, format='png')
plt.show()

print("May the force be with you!")