import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from src.utils import util_data

def plot_training(history, plot_training_dir):
    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()
    if 'loss_gen' in history.columns and 'loss_disc' in history.columns:
        plt.figure(figsize=(8, 6))
        for c, color, in zip(['loss_gen', 'loss_disc'],['r', 'b']):
            plt.plot(history[c], label=c, color=color)
        plt.title("Training G and D loss")
        plt.xlabel('Steps')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, "train_loss.png"), dpi=400, format='png')
        #plt.show()
    if 'fid' in history.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(history['fid'], label='fid', color='r')
        plt.title("Frechet Inception Distance")
        plt.xlabel('epochs')
        plt.ylabel('fid')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, "fid.png"), dpi=400, format='png')
        #plt.show()
    if 'is' in history.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(history['is'], label='is', color='r')
        plt.title("Inception Score")
        plt.xlabel('epochs')
        plt.ylabel('is')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, "is.png"), dpi=400, format='png')
        #plt.show()
    if 'rec_loss_syn' in history.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(history['rec_loss_syn'], label='reconstruction loss', color='r')
        plt.title("Reconstruction Loss Synthetic Samples")
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(plot_training_dir, "rec_loss_synthetic.png"), dpi=400, format='png')
        # plt.show()

def save_synthetic_images(general_reports_dir, epoch, noise, generator, img_dim=None, channel=None, side=None):
    images = generator(noise)[:16]


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
