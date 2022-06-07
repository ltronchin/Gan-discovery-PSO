import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

import torchvision
import torch
import torchvision.transforms.functional as F

from src.utils import  util_data

def plot_training(history, plot_training_dir):
    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    if f'train_loss_recons' in history.columns and f'val_ood_loss_recons' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip([f'train_loss_recons', f'val_ood_loss_recons'],['r', 'b']):
            plt.plot(history[c], label=c, color=color)
        plt.title(f"Reconstruction Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"reconstruction_loss.png"), dpi=400, format='png')
        #plt.show()

    if f'train_loss_vq' in history.columns and f'val_ood_loss_vq' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip([f'train_loss_vq', f'val_ood_loss_vq'],['r', 'b']):
            plt.plot(history[c], label=c, color=color)
        plt.title(f"vq loss")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"vq_loss.png"), dpi=400, format='png')
        #plt.show()

def save_image(path, image):
  """Saves an image to disk.
  NOTE: The input image (if colorful) is assumed to be with `RGB` channel order
  and pixel range [0, 255].
  Args:
    path: Path to save the image to.
    image: Image to save.
  """
  if image is None:
    return
  assert len(image.shape) == 2
  cv2.imwrite(path, image)


def show_gan_images(general_reports_dir, epoch, noise, decoder, img_dim=None, channel=None, side=None):
    with torch.no_grad():
        images = decoder(noise)
    if side is None:
        side = util_data.round_half_up(np.sqrt(images.shape[0]))
    if img_dim is None:
        img_dim = images.shape[2]
    if channel is None:
        channel = images.shape[1]

    if images.shape[-1] == img_dim:  # convert to channel last
        images = np.transpose(images.cpu().detach().numpy(), (0, 2, 3, 1))

    if images.shape[0] < side * side:
        diff = side * side - images.shape[0]
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

    superimg = (superimg * 0.5 + 0.5) * 255  # rescale from [-1, 1] to [0, 255]
    superimg = np.clip(superimg, 0, 255)  # clip the values outside the range [0, 255] to [0, 255]
    if channel == 3:
        gray_image = cv2.cvtColor(superimg.astype('float32'), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(general_reports_dir, f'synthetic_images_{epoch}.png'), gray_image)
    else:
        cv2.imwrite(os.path.join(general_reports_dir, f'synthetic_images_{epoch}.png'), superimg)

def show_images(general_reports_dir, epoch, data_loaders, model, device, phase, n_img=10):
    dataset = data_loaders.dataset
    plt.figure(figsize=(9, 2))
    for i in range(n_img):
        img = dataset[i][0].unsqueeze(0)

        with torch.no_grad():
            img = img.to(device)
            rec_img, _, _ = model(img)

        ax = plt.subplot(2, n_img, i+1)
        plt.imshow(img.cpu().squeeze().detach().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)
        if i == n_img // 2:
            ax.set_title('Original images')

        ax = plt.subplot(2, n_img, n_img+i+1)
        plt.imshow(rec_img.cpu().squeeze().detach().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)
        if i == n_img // 2:
            ax.set_title('Reconstructed images')

    plt.savefig(os.path.join(general_reports_dir, f"img_loss_{phase}_{epoch + 1}.png"), dpi=400, format='png')
    #plt.show()

def show(noise, decoder):
    with torch.no_grad():
        imgs = decoder(noise)[:16]
    imgs = torchvision.utils.make_grid(imgs)

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()