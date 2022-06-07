import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from src.utils import util_data

import torchvision
import torch
import torchvision.transforms.functional as F

def plot_training(history, plot_training_dir, phase='train'):
    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    if 'train_loss' in history.columns and 'val_iid_loss' in history.columns and 'val_ood_loss' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip(['train_loss', 'val_iid_loss', 'val_ood_loss'],['r', 'b', 'g']):
            plt.plot(history[c], label=c, color=color)
        plt.title("Training and validation loss")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, "train_loss.png"), dpi=400, format='png')
        #plt.show()

    if f'{phase}_loss_enc' in history.columns and f'{phase}_loss_disc' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip([f'{phase}_loss_enc', f'{phase}_loss_disc'],['r', 'b']):
            plt.plot(history[c], label=c, color=color)
        plt.title(f"{phase} G and D loss")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"{phase}_loss.png"), dpi=400, format='png')
        #plt.show()

    ### pix_rec_adv ###

    if f'{phase}_loss_enc' and f'{phase}_loss_enc_adv' and f'{phase}_loss_enc_rec_pix' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip([f'{phase}_loss_enc',  f'{phase}_loss_enc_adv', f'{phase}_loss_enc_rec_pix'], ['r', 'b', 'g']):
            plt.plot(history[c], label=c, color=color)
        plt.title(f"{phase} G losses")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"{phase}_G_losses.png"), dpi=400, format='png')
        # plt.show()

    if f'{phase}_loss_disc' and f'{phase}_loss_disc_adv' and f'{phase}_loss_disc_r1penalty' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip([f'{phase}_loss_disc', f'{phase}_loss_disc_adv', f'{phase}_loss_disc_r1penalty'], ['r', 'b', 'g']):
            plt.plot(history[c], label=c, color=color)
        plt.title(f"{phase} D losses")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"{phase}_D_losses.png"), dpi=400, format='png')
        # plt.show()

    ### pix_fea_rec_adv ###
    if f'{phase}_loss_enc' and f'{phase}_loss_enc_adv' and f'{phase}_loss_enc_rec_pix' and f'{phase}_loss_enc_rec_fea' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip([f'{phase}_loss_enc',  f'{phase}_loss_enc_adv', f'{phase}_loss_enc_rec_pix', f'{phase}_loss_enc_rec_fea'], ['r', 'b', 'g', 'm']):
            plt.plot(history[c], label=c, color=color)
        plt.title(f"{phase} G losses")
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, f"{phase}_G_losses.png"), dpi=400, format='png')
        # plt.show()

    # Regularize inverter
    if 'loss_pix' in history.columns and 'loss_reg' and 'loss' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip(['loss_pix', 'loss_reg', 'loss'], ['r', 'b', 'g']):
            plt.plot(history[c], label=c, color=color)
        plt.title("Optimization losses")
        plt.xlabel('Iterations')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, "regularize_inverter_losses.png"), dpi=400, format='png')
        # plt.show()

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

def show_cae_images(general_reports_dir, epoch, data_loaders, phase, encoder, decoder, device, n_img=10):
    dataset = data_loaders[phase].dataset
    plt.figure(figsize=(9, 2))
    for i in range(n_img):
        img = dataset[i][0].unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            rec_img = decoder(encoder(img))

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

    plt.savefig(os.path.join(general_reports_dir, f"img_loss_{phase}_{epoch}.png"), dpi=400, format='png')
    plt.show()

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