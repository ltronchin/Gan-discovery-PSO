# -----------------------------------
# Author
#   Lorenzo Tronchin
# Script description
#   module defining dcgan
# Date
#
# -----------------------------------

import torch.nn as nn
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchsummary import summary

from tqdm import tqdm
import time
import os
import pickle
import copy
import numpy as np

from src.utils import util_report_gan
from src.utils import util_data
from src.evaluation import util_cae, util_classifiers, util_gan_evaluation


def load_gan(model_dir, z_dim, channel, units_gen, device):
    generator = Generator(z_dim=z_dim, channels_img=channel,  features_g=units_gen)
    checkpoint_g = torch.load(os.path.join(model_dir, 'checkpoint_g.tar'), map_location=device)
    generator.load_state_dict(checkpoint_g['model_state_dict'])
    generator.to(device)
    return generator

def get_opti(model_parameters, name, lr, weight_decay=0, beta1=0.9, beta2=0.999, epsilon=0.00000001):
    if name == 'Adam':
        return optim.Adam(params=model_parameters, lr=lr, betas=(beta1, beta2),  eps=epsilon, weight_decay=weight_decay)
    elif name == 'RMSprop':
        return optim.RMSprop(params=model_parameters, lr=lr, eps=epsilon, weight_decay=weight_decay)
    else:
        raise ValueError(name)

# Function to initialize the weight of a particular model
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # according to the DCGAN paper

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # according to the DCGAN paper
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


def sanity_check(z_dim, device, n_img=8, image_size=112, channels_img=1):
    x = torch.randn((n_img, channels_img, image_size, image_size)).to(device)  # define a random tensor of the same dimension of input tensor
    disc = Discriminator(channels_img=channels_img, features_d=64)
    disc.to(device)
    summary(disc,  (channels_img, image_size, image_size))# define the discriminator network
    initialize_weights(disc)
    assert disc(x).shape == (n_img, 1, 1, 1)  # shape of the output of the discriminator

    gen = Generator(z_dim=z_dim, channels_img=channels_img, features_g=64)
    gen.to(device)
    initialize_weights(gen)
    z = torch.randn((n_img, z_dim, 1, 1)).to(device)
    assert gen(z).shape == (n_img, channels_img, image_size, image_size)
    summary(gen,  (z_dim, 1, 1))

    print("Success")

def label_smoothing(y, device):
    # positive label smoothing -->  class=1 to [0.7, 1.2]
    # negative label smoothing -->  class=0 to [0.0, 0.3]
    if y[0].item() == 1:
        return y - 0.3 + (torch.rand(y.shape).to(device) * 0.5)
    else:
        return y + torch.rand(y.shape).to(device) * 0.3

def noisy_labels():
    pass

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        # in the original DCGAN implementation, the batch normalization is not used in the first layer of the discriminator and in the last layer of the generator
        self.disc = nn.Sequential(
            # Input: N x channels_img, 28, 28
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2),
            self._block(in_channels=features_d, out_channels=features_d * 2, kernel_size=4, stride=2, padding=1), # 7x7
            nn.Conv2d(in_channels=features_d * 2, out_channels=1, kernel_size=7, stride=2, padding=0),
            nn.Sigmoid()  # the output is a single value representing the probability of the image to be real
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):  # z_dim represent the noise dimension
        super(Generator, self).__init__()

        # in the original DCGAN implementation, the batch normalization is not used in the first layer of the discriminator and in the last layer of the generator
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(in_channels=z_dim, out_channels=features_g * 2, kernel_size=7, stride=1, padding=0), # 7 x 7
            self._block(in_channels=features_g * 2, out_channels=features_g, kernel_size=4, stride=2, padding=1), # 14 x 14
            nn.ConvTranspose2d(in_channels=features_g, out_channels=channels_img, kernel_size=4, stride=2, padding=1), # 28 x 28
            nn.Tanh(), # [-1, 1] if the real images are rescaled in [-1 1] also the synthetic images must be rescaled in the same range
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), # we neeed to upsample
        )

    def forward(self, x):
        return self.gen(x)

def train(general_reports_dir, plot_training_dir, model_dir, epochs, device, loader, z_dim, batch_size, image_size,
          generator, discriminator, criterion, optimizer_gen, optimizer_disc, y_smoothing, resume_training,
          encoder, decoder, noise_factor, classifiers, val_loader):
    since = time.time()

    writer_loss = SummaryWriter(general_reports_dir + "/logs/loss")
    writer_evaluation = SummaryWriter(general_reports_dir + "/logs/evaluation")
    writer_real = SummaryWriter(general_reports_dir + "/logs/real")  # create a SummaryWriter instance.
    writer_fake = SummaryWriter(general_reports_dir + "/logs/fake")
    fixed_noise = torch.randn(32, z_dim, 1, 1).to(device) # returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1

    # Resume training
    if resume_training == '.tar':
        print('Resume training')
        checkpoint_g = torch.load(os.path.join(model_dir, 'checkpoint_g.tar'))
        generator.load_state_dict(checkpoint_g['model_state_dict'])
        optimizer_gen.load_state_dict(checkpoint_g['optimizer_state_dict'])

        checkpoint_d = torch.load(os.path.join(model_dir, 'checkpoint_d.tar'))
        discriminator.load_state_dict(checkpoint_d['model_state_dict'])
        optimizer_disc.load_state_dict(checkpoint_d['optimizer_state_dict'])
        offset = checkpoint_g['epoch'] + 1
        with open(os.path.join(general_reports_dir, 'history.pkl'), 'rb') as f:
            history = pickle.load(f)
    else:
        history = {'loss_gen': [], 'loss_disc': [], 'fid': [], 'is': [], 'rec_loss_syn': []}
        offset = 0

    best_generator_wts = copy.deepcopy(generator.state_dict())
    best_discriminator_wts = copy.deepcopy(discriminator.state_dict())
    best_loss = 0.0
    best_epoch = offset

    for epoch in range(epochs):
        print(f"\nTRAIN MODEL, epoch: {epoch}")
        generator.train()
        discriminator.train()
        #with tqdm(total=len(loader.dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for real, _ in loader: # GANs are unsupervised!
            real = real.to(device)
            noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)

            fake = generator(noise)  # generate fake images feeding the discriminator with noise

            # Train Discriminator: minimize [-log(D(x)) - log(1 - D(G(z)))] (maximize[log(D(x)] is equal to say "maximize the probability of real images being real)
            disc_real = discriminator(real).reshape(-1)  # we reshape the output N x 1 x 1 x 1 to get only the N predicted values
            disc_fake = discriminator(fake).reshape(-1)

            y_real = torch.ones_like(disc_real).to(device)
            y_fake = torch.zeros_like(disc_fake).to(device)
            if y_smoothing is not None:
                y_real = label_smoothing(y_real, device)
                y_fake = label_smoothing(y_fake, device)

            loss_disc_real = criterion(disc_real, y_real)  # when we compute BCE and y=1 only survive the first term in the loss -log(D(x))
            loss_disc_fake = criterion(disc_fake, y_fake)  # when we compute BCE and y=0 only survive the second term in the loss - log(1 - D(G(z)))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2  # divide by two as the discriminator see 2 mini-batches with respect to the generator that sees 1 minibatch
            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)  # retain_graph --> keep all variables or flags associated with computed gradients. We set retain_graph to true
            # as we are gonna to reuse the gradient information when fake images are passed through the discriminator
            optimizer_disc.step()

            # Train Generator: minimize [log(1 - D(G(z))] < -- > maximize [log(D(G(z)))]  < -- > minimize [-log(D(G(z)))] (remember the saturating problem)
            output = discriminator(fake).reshape(-1)  # compute the gradient
            loss_gen = criterion(output, y_real)
            generator.zero_grad()
            loss_gen.backward() # only modify the generator weight
            optimizer_gen.step()

            history['loss_gen'].append(loss_gen.item())
            history['loss_disc'].append(loss_disc.item())
            #pbar.set_postfix(**{"Loss D": loss_disc.item(), "loss G": loss_gen.item()})
            #pbar.update(real.shape[0])

        # Checkpoint at the end of every epoch
        print(f"Save model at epoch: {epoch}")
        torch.save(
            {'epoch': epoch + offset,
             'model_state_dict': generator.state_dict(),
             'optimizer_state_dict': optimizer_gen.state_dict(),
             'loss': loss_gen.item()}, os.path.join(model_dir, "checkpoint_g.tar")
        )
        torch.save(
            {'epoch': epoch + offset,
             'model_state_dict': discriminator.state_dict(),
             'optimizer_state_dict': optimizer_disc.state_dict(),
             'loss': loss_disc.item()}, os.path.join(model_dir, "checkpoint_d.tar")
        )

        print(f"\nEVALUATION PHASE, epoch: {epoch}")
        # Create embeddings with cae
        print("Create embeddings")
        syn_dataset = util_data.DatasetSyntheticImg(model=generator, z_dim=z_dim, image_size=image_size, max_len=batch_size*100, device=device) #len(val_loader.dataset)
        syn_loader = torch.utils.data.DataLoader(dataset=syn_dataset, batch_size= batch_size, shuffle=True)
        # Create encoded sample for validation set
        encoded_real, sample_real = util_cae.create_encoded_sample_batch(encoder=encoder, dataloader=val_loader, batch_size=batch_size, device=device, n_iter=100)
        # Create encoded sample for synthetic set
        encoded_synthetic, samples_synthetic = util_cae.create_encoded_sample_batch(encoder=encoder, dataloader=syn_loader, batch_size=batch_size, device=device, n_iter=100)
        # Frechet Inception Distance
        print("Compute fid")
        mu_real, mu_synthetic, covariance_real, covariance_synthetic = util_gan_evaluation.compute_statistics(encoded_real, encoded_synthetic)
        fid = util_gan_evaluation.frechet_distance(mu_real, mu_synthetic, covariance_real, covariance_synthetic)
        history["fid"].append(fid.item())

        # Inception score
        print("Compute is")
        p_yx = util_classifiers.compute_posterior(encoding=encoded_synthetic, classifiers=classifiers)
        inception_score = util_gan_evaluation.calculate_inception_score(p_yx, eps=1E-16)
        history["is"].append(inception_score)

        # Denoising loss
        print("Compute reconstruction loss")
        loss_ae =  torch.nn.MSELoss()
        rec_loss_syn, _, _ = util_gan_evaluation.test_epoch(encoder=encoder, decoder=decoder, device=device, data=samples_synthetic, loss_fn=loss_ae, noise_factor=noise_factor)
        history['rec_loss_syn'].append(rec_loss_syn)

        # Polarization across patients
        util_gan_evaluation.plot_histogram(epoch=epoch, general_reports_dir=general_reports_dir, p_yx=p_yx, variables=['energy', 'variance'])
        util_gan_evaluation.plot_posterior_polarization(epoch=epoch, general_reports_dir=general_reports_dir, p_yx=p_yx)

        print(f"Epoch: {epoch}, fid: {fid.item()}, is: {inception_score}, rec_loss_syn: {rec_loss_syn}")
        # Save history, training plot, synthetic images and the evaluation metrics
        with open(os.path.join(general_reports_dir, 'history_gan.pkl'), 'wb') as f:
            pickle.dump(history, f)
        util_report_gan.plot_training(history, plot_training_dir)
        util_report_gan.save_synthetic_images(general_reports_dir, epoch + offset, fixed_noise, generator, img_dim=28, channel=1)

        # Check for the best model
        if inception_score > best_loss:
            best_epoch = epoch
            best_loss = inception_score
            best_generator_wts = copy.deepcopy(generator.state_dict())
            best_discriminator_wts = copy.deepcopy(discriminator.state_dict())

        generator.eval()
        with torch.no_grad():  # plot same fake images to tensorboard (no gradient needed)
            fake = generator(fixed_noise)
            # take out (up to) 32 examples
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

            writer_real.add_image("Real", img_grid_real, global_step = epoch + offset)
            writer_fake.add_image("Fake", img_grid_fake, global_step = epoch + offset)
            writer_loss.add_scalar('gen', loss_gen.item(), global_step = epoch + offset)
            writer_loss.add_scalar('disc', loss_disc.item(), global_step = epoch + offset)
            writer_evaluation.add_scalar('fid', fid.item(), global_step = epoch + offset)
            writer_evaluation.add_scalar('is', inception_score, global_step=epoch + offset)

    generator.load_state_dict(best_generator_wts)
    discriminator.load_state_dict(best_discriminator_wts)

    print(f"Save the best model finded during training: {best_epoch}")
    torch.save(
        {'epoch': best_epoch + offset,
         'model_state_dict': generator.state_dict(),
         'optimizer_state_dict': optimizer_gen.state_dict(),
         'loss': loss_gen.item()}, os.path.join(model_dir, "best_g.tar")
    )
    torch.save(
        {'epoch': best_epoch + offset,
         'model_state_dict': discriminator.state_dict(),
         'optimizer_state_dict': optimizer_disc.state_dict(),
         'loss': loss_disc.item()}, os.path.join(model_dir, "best_d.tar")
    )

    # End of training
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))