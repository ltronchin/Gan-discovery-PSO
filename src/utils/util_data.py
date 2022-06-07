
import numpy as np
import scipy.io as sio
import cv2
import os
from PIL import Image
import random
import math
import itertools
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from src.utils import util_general
from src.utils import util_mnist

def get_dataset(dataset_name, data, classes, data_dir, cfg_data, geometrical_augmentation, zoom_aug, elastic_aug, step):
    if dataset_name == "claro":
        return DatasetSlidingWindowClaro(data, classes, data_dir, cfg_data, geometrical_augmentation, zoom_aug, elastic_aug, step)
    elif dataset_name == "overall_survival":
        return DatasetSlidingWindowAerts(data, classes, data_dir, cfg_data, geometrical_augmentation, zoom_aug, elastic_aug, step)
    else:
        raise ValueError(dataset_name)

def get_public_dataset_inverter(dataset_name, data_dir, drange_net, general_reports_dir, image_size, channel, iid_class):
    if dataset_name == "mnist":
        if drange_net[0] == 0 and drange_net[1] == 1:
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            )
            val_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            )
        elif drange_net[0] == -1 and drange_net[1] == 1: # dcgan scale
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5 for _ in range(channel)], [0.5 for _ in range(channel)]),
                ]
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5 for _ in range(channel)], [0.5 for _ in range(channel)]),
                ]
            )
        else:
            raise ValueError(drange_net)

        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True) # download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        val_dataset = datasets.MNIST(root=data_dir, train=False, transform=val_transform, download=True) # train=False to download the test dataset
        if len(iid_class) != len(np.unique(train_dataset.targets)):
            train_dataset = util_mnist.split_MNIST(dataset=train_dataset, iid_digits=iid_class)
            val_dataset = util_mnist.split_MNIST(dataset=val_dataset, iid_digits=iid_class)

        util_mnist.plot_digits(train_dataset, general_reports_dir)
        return train_dataset, val_dataset

def get_public_dataset(dataset_name, data_dir, drange_net, general_reports_dir, image_size, channel, iid_class):
    if dataset_name == "mnist":
        if drange_net[0] == 0 and drange_net[1] == 1:
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ]
            )
        elif drange_net[0] == -1 and drange_net[1] == 1: # dcgan scale
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5 for _ in range(channel)], [0.5 for _ in range(channel)]),
                ]
            )
        else:
            raise ValueError(drange_net)

        val_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor()
            ]
        )

        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True) # download (bool, optional) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        val_dataset = datasets.MNIST(root=data_dir, train=False, transform=val_transform, download=True) # train=False to download the test dataset
        if len(iid_class) != len(np.unique(train_dataset.targets)):
            train_dataset = util_mnist.split_MNIST(dataset=train_dataset, iid_digits=iid_class)
            val_dataset = util_mnist.split_MNIST(dataset=val_dataset, iid_digits=iid_class)

        util_mnist.plot_digits(train_dataset, general_reports_dir)
        return train_dataset, val_dataset

    elif dataset_name == "celeba":
        pass
    elif dataset_name == "cifar":
        pass
    else:
        raise ValueError(dataset_name)

def adjust_dynamic_range(data, drange_in, drange_out): # drange_in > data range of input data, drange_out > data range to be inputted to the newtork
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def rescale_torch(img, min_val=None, max_val=None):
    if not min_val:
        min_val = torch.min(img)
    if not max_val:
        max_val = torch.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img

def rescale(img, min_val=None, max_val=None):
    if not min_val:
        min_val = np.min(img)
    if not max_val:
        max_val = np.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img

def load_img(img_path):
    filename, extension = os.path.splitext(img_path)
    if extension == ".mat":
        load = sio.loadmat(img_path)
        img = np.asarray(load['img'], dtype = 'float32')
    else:
        img = Image.open(img_path)
        img = np.asarray(img, dtype = 'float32')
    return img


def elastic_transform(img, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    Arguments
       img: Numpy array with shape (height, width).
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """
    assert len(img.shape) == 2
    shape = img.shape

    if random_state is None:
        random_state = np.random.RandomState(None)
    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    return map_coordinates(img, indices, order=1, mode='constant', cval=0.0).reshape(shape)

def clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def cv2_rotate(img, angle, center = None, scale = 1.0):
    (h, w) = img.shape[:2] # the size of the output image is the same of the original one
    if center is None:
        center = (w / 2, h / 2) # we set the centre of rotation as the centre of the input image
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale) # create the rotation matrix
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,  borderValue=0.0)
    return rotated

def cv2_shift(img, shiftX, shiftY):
    (h, w) = img.shape[:2]  # the size of the output image is the same of the original one
    M = np.float32([
        [1, 0, shiftX],
        [0, 1, shiftY]
    ])
    shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return shifted

def augmentation(img, zoom_aug, elastic_aug):
    # horizontal flip
    r = random.randint(0, 100)
    if r > 70:
        img = cv2.flip(img, 1)

    # vertical flip
    r = random.randint(0, 100)
    if r > 70:
        cv2.flip(img, 0)

    # shift
    r = random.randint(0, 100)
    if r > 70:
        shift_perc = 0.1
        r1 = random.randint(-int(shift_perc*img.shape[0]), int(shift_perc*img.shape[0]))
        r2 = random.randint(-int(shift_perc*img.shape[1]), int(shift_perc*img.shape[1]))
        img = cv2_shift(img, r1, r2)

    # rotation
    r = random.randint(0, 100)
    if r > 70:
        max_angle = 175
        r = random.randint(-max_angle, max_angle)
        img = cv2_rotate(img, r)

    # zoom
    if zoom_aug:
        # print('zoom aug')
        r = random.randint(0, 100)
        if r > 70:
            zoom_perc = 0.1
            zoom_factor = random.uniform(1 - zoom_perc, 1 + zoom_perc)
            img = clipped_zoom(img, zoom_factor=zoom_factor)

    # elastic deformation
    if elastic_aug:
        # print('elastic aug')
        r = random.randint(0, 100)
        if r > 70:
            img = elastic_transform(img, alpha_range=[20, 40], sigma=7)
    return img

def loader(img_path, img_dim, step="train", rescale_minus_1_plus_1=None, geometrical_augmentation=None, zoom_aug=False, elastic_aug=False, fill_nan=-1000):
    # Load image
    img = load_img(img_path) # return N x N float32 numpy image

    # Remove nan
    if np.isnan(img).any(): # remove nan
        mask = np.isnan(img)
        img[mask] = fill_nan
    assert ~np.isnan(img).any()

    # Resize
    if img.shape[0] != img_dim:
        img = cv2.resize(img, (img_dim, img_dim))

    # Rescale
    min_val, max_val = np.min(img), np.max(img)
    img = rescale(img, min_val=min_val, max_val=max_val)

    # Augmentation
    if step == "train" and geometrical_augmentation:
        img = augmentation(img, zoom_aug, elastic_aug)

    # Rescale in [-1, 1]
    if rescale_minus_1_plus_1:
        img = (img - 0.5) * 2

    # To Tensor
    img = torch.from_numpy(img)

    # Add dimension
    if img.ndim < 3:
        img = torch.unsqueeze(img, dim=0)
    return img


def round_half_up(number, decimals=0):
    multiplier = 10 ** decimals
    return int(math.ceil(number * multiplier + 0.5) / multiplier)

def save_dataset_images(reports_dir, data_loaders, img_dim, step, channel=1, side=None):
    images = []
    for x_batch, y_batch, id_batch, id_slices_batch in data_loaders:
        images.append(x_batch.cpu().detach().numpy())
    images = np.asarray(list(itertools.chain(*images)))

    if images.shape[-1] == img_dim:  # convert to channel last
        images = np.transpose(images, (0, 2, 3, 1))

    if side is None:
        side = round_half_up(np.sqrt(images.shape[0]))

    if images.shape[0] < side * side:
        diff = side * side - images.shape[0]
        # print('[INFO] Adding, ' + str(diff) + ' blank images.')
        blank_imgs = np.zeros((diff, img_dim, img_dim, channel))
        imgs = np.concatenate((images, blank_imgs))

    superimg = np.zeros(shape=(img_dim, img_dim, channel))
    for index in range(0, side):
        img_concat_row = imgs[index * side][:, :, :]
        for i in range((index * side) + 1, (index + 1) * side):
            img = imgs[i]
            img_concat_row = cv2.hconcat([img_concat_row, img])

        if superimg.shape[1] == img_dim:
            superimg = img_concat_row
        else:
            superimg = cv2.vconcat([superimg, img_concat_row])
    if channel == 3:
        gray_image = cv2.cvtColor(superimg.astype('float32'), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(reports_dir, step + '_superimg.png'), gray_image * 255)
    else:
        cv2.imwrite(os.path.join(reports_dir, step + '_superimg.png'), superimg * 255)

def plot_generated(examples, n):
    for i in range(n):
        plt.subplot(1, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :],  cmap='gray')
    plt.show()

class DatasetSlidingWindowClaro(torch.utils.data.Dataset):
    """ Characterizes a dataset for PyTorch Dataloader to train images """
    def __init__(self, data, classes, data_dir, cfg_data, geometrical_augmentation, zoom_aug, elastic_aug, step):
        """ Initialization """
        self.data = data
        self.classes = classes
        self.img_dir = util_general.create_path(data_dir, cfg_data['channel'], cfg_data['image_size'], cfg_data['nan_cutoff'])
        self.img_dim = cfg_data['image_size']
        self.geo_aug = geometrical_augmentation
        self.zoom_aug = zoom_aug
        self.elastic_aug = elastic_aug
        self.step = step
        self.rescale_minus_1_plus_1 = cfg_data['rescale_minus_1_plus_1']

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.data)

    def __getitem__(self, index):
        """ Generates one sample of data """
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        id_slice = row.id_slice
        img_name = id + '_' + str(id_slice)

        # Load data and get label
        img_path = os.path.join(self.img_dir, img_name+'.mat')
        x = loader(img_path=img_path, img_dim=self.img_dim, step=self.step, rescale_minus_1_plus_1=self.rescale_minus_1_plus_1, geometrical_augmentation= self.geo_aug, zoom_aug=self.zoom_aug, elastic_aug=self.elastic_aug)
        y = row.label
        return x, y, id, id_slice

class DatasetSlidingWindowAerts(torch.utils.data.Dataset):
    """ Characterizes a dataset for PyTorch Dataloader to train images """
    def __init__(self, data, classes, data_dir, cfg_data, geometrical_augmentation, zoom_aug, elastic_aug, step):
        """ Initialization """
        self.data = data
        self.classes = classes
        self.img_dir = util_general.create_path(data_dir, cfg_data['channel'], cfg_data['image_size'])
        self.img_dim = cfg_data['image_size']
        self.geo_aug = geometrical_augmentation
        self.zoom_aug = zoom_aug
        self.elastic_aug = elastic_aug
        self.step = step
        self.rescale_minus_1_plus_1 = cfg_data['rescale_minus_1_plus_1']

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.data)

    def __getitem__(self, index):
        """ Generates one sample of data """
        # Select sample
        row = self.data.iloc[index]
        id = row.name
        id_slice = row.id_slice
        img_name = id + '_' + str(id_slice)

        # Load data and get label
        img_path = os.path.join(self.img_dir, img_name+'.mat')
        x = loader(img_path=img_path, img_dim=self.img_dim, step=self.step, rescale_minus_1_plus_1=self.rescale_minus_1_plus_1, geometrical_augmentation= self.geo_aug, zoom_aug=self.zoom_aug, elastic_aug=self.elastic_aug)
        y = row.label
        return x, y, id, id_slice

class DatasetSyntheticImg(torch.utils.data.Dataset):
    """ Characterizes a dataset for PyTorch Dataloader to train images """
    def __init__(self, model, z_dim, image_size, max_len, device):
        """ Initialization """
        self.model = model
        self.z_dim = z_dim
        self.img_dim = image_size
        self.device = device
        self.max_len = max_len

    def __len__(self):
        """ Denotes the total number of samples """
        return self.max_len

    def __getitem__(self, index):
        """ Generates one sample of data """
        self.model.eval()
        noise = torch.randn((1, self.z_dim, 1, 1)).to(self.device)
        with torch.no_grad():
            img = self.model(noise)

        img = rescale_torch(img)

        return img[0], math.nan

class DatasetSyntheticImgPSO(torch.utils.data.Dataset):
    """ Characterizes a dataset for PyTorch Dataloader to train images """
    def __init__(self, model, z_dim, image_size, max_len, device):
        """ Initialization """
        self.model = model
        self.z_dim = z_dim
        self.img_dim = image_size
        self.device = device
        self.max_len = max_len

    def __len__(self):
        """ Denotes the total number of samples """
        return self.max_len

    def __getitem__(self, index):
        """ Generates one sample of data """
        self.model.eval()
        noise = torch.randn((1, self.z_dim, 1, 1)).to(self.device)
        with torch.no_grad():
            img = self.model(noise)

        img = rescale_torch(img)

        return img[0], math.nan