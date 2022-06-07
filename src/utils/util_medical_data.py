import torch
import numpy as np
from PIL import Image
import cv2
import os
import random
import math
import pandas as pd
import re

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_img(img_path):
    filename, extension = os.path.splitext(img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(float)
    return img


def get_box(img, box, perc_border=.0):
    # Sides
    l_h = box[2] - box[0]
    l_w = box[3] - box[1]
    # Border
    diff_1 = math.ceil((abs(l_h - l_w) / 2))
    diff_2 = math.floor((abs(l_h - l_w) / 2))
    border = int(perc_border * diff_1)
    # Img dims
    img_h = img.shape[0]
    img_w = img.shape[1]
    if l_h > l_w:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-diff_1-border < 0:
            pad = 0-(box[1]-diff_1-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+diff_2+border > img_w:
            pad = (box[3]+diff_2+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-diff_1-border:box[3]+diff_2+border]
    elif l_w > l_h:
        if box[0]-diff_1-border < 0:
            pad = 0-(box[0]-diff_1-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+diff_2+border > img_h:
            pad = (box[2]+diff_2+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-diff_1-border:box[2]+diff_2+border, box[1]-border:box[3]+border]
    else:
        if box[0]-border < 0:
            pad = 0-(box[0]-border)
            img = np.vstack([np.zeros((pad, img.shape[1])), img])
            box[0] += pad
            box[2] += pad
            img_h += pad
        if box[2]+border > img_h:
            pad = (box[2]+border)-img_h
            img = np.vstack([img, np.zeros((pad, img.shape[1]))])
        if box[1]-border < 0:
            pad = 0-(box[1]-border)
            img = np.hstack([np.zeros((img.shape[0], pad)), img])
            box[1] += pad
            box[3] += pad
            img_w += pad
        if box[3]+border > img_w:
            pad = (box[3]+border)-img_w
            img = np.hstack([img, np.zeros((img.shape[0], pad))])
        img = img[box[0]-border:box[2]+border, box[1]-border:box[3]+border]
    return img


def normalize(img, min_val=None, max_val=None):
    if not min_val:
        min_val = img.min()
    if not max_val:
        max_val = img.max()
    img = (img - min_val) / (max_val - min_val)
    # img -= img.mean()
    # img /= img.std()
    return img

def loader(img_path, img_dim, box=None, clip=None, scale=None, clahe=False):
    # Img
    img = load_img(img_path)
    min_val, max_val = img.min(), img.max()

    # Select Box Area
    if box:
        img = get_box(img, box, perc_border=0.5)
    # Resize
    img = cv2.resize(img, (img_dim, img_dim))
    # Clip
    if clip:
        img = np.clip(img, clip['min'], clip['max'])
    # Normalize
    if scale:
        img = normalize(img, min_val=scale['min'], max_val=scale['max'])
    else:
        img = normalize(img, min_val=min_val, max_val=max_val)

    # To Tensor
    img = torch.Tensor(img)
    img  = img.unsqueeze(0)
    return img

class ImgDatasetPreparation(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, cfg_data):
        """Initialization"""
        self.img_dir = os.path.join(cfg_data["data_dir"], cfg_data["dataset"])
        self.data = data # estrai da file patients_info_CLARO_retrospettivo.xlsx
        # Box
        if cfg_data["box_file"]:
            box_data = pd.read_excel(cfg_data["box_file"], index_col="img ID", dtype=list)
            self.boxes = {os.path.basename(row[0]): eval(row[1][cfg_data["box_value"]]) for row in box_data.iterrows()}
        else:
            self.boxes = None
        # Clip
        self.clip = cfg_data["clip"]
        # Scale
        self.scale = cfg_data["scale"]
        # Dim
        self.img_dim = cfg_data["image_size"]

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        row = self.data.iloc[index].split('_')
        img_id = row[1]
        patient_id = row[0]

        # load box
        if self.boxes:
            box = self.boxes[patient_id + '_' + img_id]
        else:
            box = None
        # Load data and get label
        img_path = os.path.join(self.img_dir, patient_id, "images", f"{patient_id + '_' + img_id}.tif")
        x = loader(img_path=img_path, img_dim=self.img_dim, box=box, clip=self.clip, scale=self.scale)

        return x, patient_id, img_id
