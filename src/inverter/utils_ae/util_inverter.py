import time
import copy
import numpy as np
import pandas as pd
import os
import pickle
#from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torchsummary import summary

from src.inverter.utils_ae import util_report_inverter
from src.inverter.utils_ae import util_nn
from src.utils import util_general

def freeze_parameters(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def get_opti(model_parameters, name, lr, weight_decay=0, beta1=0.9, beta2=0.999, epsilon=0.00000001):
    if name == 'Adam':
        return optim.Adam(params=model_parameters, lr=lr, betas=(beta1, beta2),  eps=epsilon, weight_decay=weight_decay)
    elif name == 'RMSprop':
        return optim.RMSprop(params=model_parameters, lr=lr, eps=epsilon, weight_decay=weight_decay)
    else:
        raise ValueError(name)

def get_train_fun(training_fun, cfg, general_reports_dir, plot_training_dir, model_dir, epochs, data_loaders,latent_dim, encoder, decoder, device, discriminator=None, multipatient_cnn=None):
    if training_fun == "pix_rec":
        print(training_fun)

        decoder = freeze_parameters(model=decoder)
        optimizer = get_opti(encoder.parameters(), **cfg['trainer_inverter']['discriminator_optimizer'])
        rec_criterion = torch.nn.MSELoss().to(device)
        return train_pix_rec(general_reports_dir=general_reports_dir,
                         plot_training_dir=plot_training_dir,
                         model_dir=model_dir,
                         epochs=epochs,
                         data_loaders=data_loaders,
                         latent_dim=latent_dim,
                         encoder=encoder,
                         decoder=decoder,
                         device=device,
                         optimizer=optimizer,
                         criterion=rec_criterion)

    elif training_fun == "pix_fea_rec":
        # todo pix_fea_rec
        raise NotImplementedError

    elif training_fun == "pix_rec_adv":
        # todo pix_rec_adv
        raise NotImplementedError

    elif training_fun == "pix_fea_rec_adv":
        print(training_fun)
        if discriminator is None:
            raise ModuleNotFoundError
        if multipatient_cnn is None:
            raise NotImplementedError
        decoder = freeze_parameters(model=decoder)
        multipatient_cnn = freeze_parameters(model=multipatient_cnn)
        optimizer_E = get_opti(encoder.parameters(), **cfg['trainer_inverter']['discriminator_optimizer'])
        optimizer_D = get_opti(discriminator.parameters(), **cfg['trainer_inverter']['discriminator_optimizer'])

        rec_criterion = torch.nn.MSELoss().to(device)
        adv_criterion = nn.BCELoss().to(device)
        return train_pix_fea_rec_adv(general_reports_dir=general_reports_dir,
                             plot_training_dir=plot_training_dir,
                             model_dir = model_dir,
                             epochs=epochs,
                             data_loaders=data_loaders,
                             latent_dim=latent_dim,
                             encoder=encoder,
                             decoder=decoder,
                             device=device,
                             discriminator=discriminator,
                             net=multipatient_cnn,
                             opt_E=optimizer_E,
                             opt_D=optimizer_D,
                             rec_crit=rec_criterion,
                             adv_crit=adv_criterion)
    else:
        raise ValueError(training_fun)

# Function to initialize the weight of a particular model
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)  # according to the DCGAN paper

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

class Encoder_AttGAN(nn.Module):
    def __init__(self, enc_dim, channels_img, features_e=16, enc_layers= 4, enc_norm_fn='batchnorm', enc_acti_fn='relu'):
        super(Encoder_AttGAN, self).__init__()

        layers = []
        n_in = channels_img
        for i in range(enc_layers):
            n_out = min(features_e * 2 ** i, enc_dim)
            layers += [
                util_nn.Conv2dBlock(n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn)]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)

    def encode(self, x):
        z = x
        for layer in self.enc_layers:
            z = layer(z)
        return z

    def forward(self, x):
        return self.encode(x)

class Encoder(nn.Module):
    def __init__(self, enc_dim, channels_img, features_e=64):
        super(Encoder, self).__init__()

        self.enc = nn.Sequential(
            # Input: N x channels_img, 28, 28
            nn.Conv2d(channels_img, features_e, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2),
            self._block(in_channels=features_e, out_channels=features_e * 2, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.Conv2d(in_channels=features_e * 2, out_channels=enc_dim, kernel_size=7, stride=2, padding=0)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.enc(x)

def sanity_check(z_dim, device, n_img=8, image_size=112, channels_img=1):
    print('Check D')
    disc = Discriminator(channels_img=channels_img, features_d=64)
    disc.to(device)
    summary(disc, (channels_img, image_size, image_size), device=device)  # define the discriminator network

    initialize_weights(disc)
    x = torch.randn((n_img, channels_img, image_size, image_size)).to(device)  # define a random tensor of the same dimension of input tensor
    assert disc(x).shape == (n_img, 1, 1, 1)  # shape of the output of the discriminator


    print('Check G')
    gen = Generator(z_dim=z_dim, channels_img=channels_img, features_g=64)
    gen.to(device)
    summary(gen, (z_dim, 1, 1), device=device)
    initialize_weights(gen)
    z = torch.randn((n_img, z_dim, 1, 1)).to(device)
    assert gen(z).shape == (n_img, channels_img, image_size, image_size)

    print('Check E')
    enc = Encoder(enc_dim=z_dim, channels_img=channels_img)
    enc.to(device)
    summary(enc, (channels_img, image_size, image_size), device=device)
    x = torch.randn((n_img, channels_img, image_size, image_size)).to(device)
    assert enc(x).shape == (n_img, z_dim, 1, 1)
    print("Success")

def train_pix_rec(general_reports_dir, plot_training_dir, model_dir, epochs, data_loaders, latent_dim, encoder, decoder, optimizer, criterion, device):
    since = time.time()

    best_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_loss = np.Inf
    best_epoch = 0

    history = {'train_loss': [], 'val_iid_loss': [], 'val_ood_loss': []}
    fixed_noise = torch.randn((32, latent_dim, 1, 1)).to(device)

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch}")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val_iid', 'val_ood']:
            print(f'Phase: {phase}')
            if phase == 'train':
                encoder.train()
                decoder.eval() # Set model to training mode
            else:
                encoder.eval()
                decoder.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            #with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for img, _ in data_loaders[phase]:  # GANs are unsupervised!
                img = img.to(device)

                optimizer.zero_grad() # zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'): # forward
                    z = encoder(img.float())  # create latent representation
                    rec_img = decoder(z.float())  # synthetise from the latent representation

                    loss = criterion(img.float(), rec_img.float())
                    #pbar.set_postfix(**{'loss (batch)': loss.item()})
                    if phase == 'train':  # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # Checkpoint
                running_loss += loss.item() * img.size(0)
                #pbar.update(img.shape[0])

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            util_report_inverter.show_cae_images(general_reports_dir=general_reports_dir, epoch=epoch, data_loaders=data_loaders, phase=phase, encoder=encoder,  decoder=decoder, device=device, n_img=10)
            # Update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            elif phase == 'val_iid':
                history['val_iid_loss'].append(epoch_loss)
            elif phase == 'val_ood':
                history['val_ood_loss'].append(epoch_loss)
            else:
                raise ValueError(phase)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # Deep copy the model
            if phase == 'val_iid':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_encoder_wts = copy.deepcopy(encoder.state_dict())

        util_report_inverter.plot_training(history=history, plot_training_dir=plot_training_dir)
        util_report_inverter.show_gan_images(general_reports_dir=general_reports_dir, epoch=epoch, noise=fixed_noise, decoder=decoder)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val_ood loss: {:4f}'.format(best_loss))

    # load best model weights
    encoder.load_state_dict(best_encoder_wts)
    # Save model
    torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder.pt"))
    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()
    return encoder, history


def train_rec_fea(general_reports_dir, plot_training_dir, model_dir, epochs, data_loaders, encoder, decoder, cnn, optimizer, criterion, device):
    """
    For futher insides about implementation see functions E_loss and D_logistic_simplegp of module training/loss_encoder.py from https://github.com/genforce/idinvert
    """
    # todo train_rec_fea
    pass


def R1_reg(prediction_real: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
    # https://github.com/ChristophReich1996/Dirac-GAN
    """
    Forward pass to compute the regularization
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
        :param real_sample: (torch.Tensor) Batch of the corresponding real images
        :return: (torch.Tensor) Loss value
    """
    # Calc gradient
    gradient = torch.autograd.grad(
        outputs=prediction_real.sum(),
        inputs=real_sample,
        create_graph=True
    )[0]
    # Calc regularization
    gradient_r1_penalty = gradient.pow(2).view(gradient.shape[0], -1).sum(1).mean()
    return gradient_r1_penalty

def label_smoothing(y, device):
    # positive label smoothing -->  class=1 to [0.7, 1.2]
    # negative label smoothing -->  class=0 to [0.0, 0.3]
    if y[0].item() == 1:
        return y - 0.3 + (torch.rand(y.shape).to(device) * 0.5)
    else:
        return y + torch.rand(y.shape).to(device) * 0.3

def train_pix_fea_rec_adv(general_reports_dir, plot_training_dir, model_dir, epochs, data_loaders, latent_dim, encoder, decoder, discriminator, net, opt_E, opt_D, rec_crit, adv_crit, device, w_rec=1.0, w_fea=1.0, w_adv=0.1, r1_gamma = 10, y_smoothing=True):
    """
    For futher insides about implementation see functions E_loss and D_logistic_simplegp of module training/loss_encoder.py from https://github.com/genforce/idinvert
    """
    print(f"w_rec: {w_rec}, w_fea: {w_fea}, w_adv: {w_adv}")
    # todo far variare automaticamente i parametri w_rec, w_fea, w_adv in modo che le loss siano sempre dello stesso ordine di grandezza
    since = time.time()

    best_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_loss = np.Inf
    best_epoch = 0

    history = util_general.list_dict()
    fixed_noise = torch.randn((32, latent_dim, 1, 1)).to(device)

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch}")

        # Each epoch has a training and validation phase
        for phase in ['train', 'val_iid', 'val_ood']:
            print(f'Phase: {phase}')
            if phase == 'train':
                encoder.train() # Set model to training mode
                decoder.eval()
                discriminator.train()
                net.eval()
            else:
                encoder.eval() # Set model to evaluate mode
                decoder.eval()
                discriminator.eval()
                net.eval()

            running_loss_D, running_loss_D_adv, running_loss_D_r1penalty = 0.0, 0.0, 0.0
            running_loss_E, running_loss_E_adv, running_loss_E_rec_pix, running_loss_E_rec_fea = 0.0, 0.0, 0.0, 0.0

            #with tqdm(total=len(data_loaders[phase].dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for real, _ in data_loaders[phase]: # GANs are unsupervised!
                real = real.requires_grad_(True)
                real = real.to(device)

                # Create latent representation z
                z = encoder(real)

                fake = decoder(z)  # synthetise from the latent representation
                disc_real = discriminator(real).reshape(-1)
                disc_fake = discriminator(fake).reshape(-1)

                y_real = torch.ones_like(disc_real).to(device)
                y_fake = torch.zeros_like(disc_fake).to(device)
                if y_smoothing is not None:
                    y_real = label_smoothing(y_real, device)
                    y_fake = label_smoothing(y_fake, device)

                ############################
                # Update D network
                ###########################
                # Adversarial loss maximize log(D(x)) + log(1 - D(G(z))) -- Discriminator loss function
                loss_disc_real = adv_crit(disc_real, y_real)
                loss_disc_fake = adv_crit(disc_fake, y_fake)
                loss_D_adv = (loss_disc_real + loss_disc_fake) / 2
                # R1Penalty
                r1_penalty = R1_reg(prediction_real=disc_real, real_sample=real)
                loss_D_r1penalty = r1_penalty * (r1_gamma * 0.5)
                loss_D = loss_D_adv + loss_D_r1penalty
                if phase == 'train':
                    opt_D.zero_grad()
                    loss_D.backward(retain_graph=True)  # retain_graph --> keep all variables or flags associated with computed gradients. We set retain_graph to true as we are gonna to reuse the gradient information when fake images are passed through the discriminator
                    opt_D.step()

                ############################
                # Update E network
                ###########################
                # Pixel reconstruction loss
                loss_E_rec_pix = w_rec * rec_crit(fake, real)

                # Feature reconstruction loss (perceptual loss)
                feat_real = net.forward_avgpool(real)
                feat_fake = net.forward_avgpool(fake)
                loss_E_rec_fea = w_fea * rec_crit(feat_fake, feat_real)

                loss_E_rec = loss_E_rec_pix + loss_E_rec_fea

                # Adversarial loss maximize log(D(G(z)))
                disc_fake = discriminator(fake).reshape(-1)
                loss_E_adv = w_adv * adv_crit(disc_fake, y_real)

                loss_E = loss_E_rec + loss_E_adv
                if phase == 'train':
                    opt_E.zero_grad()
                    loss_E.backward()
                    opt_E.step()

                # Checkpoint
                running_loss_D  += loss_D.item() * real.size(0)
                running_loss_D_adv += loss_D_adv.item() * real.size(0)
                running_loss_D_r1penalty  += loss_D_r1penalty.item() * real.size(0)

                running_loss_E += loss_E.item() * real.size(0)
                running_loss_E_adv  += loss_E_adv.item() * real.size(0)
                running_loss_E_rec_pix  += loss_E_rec_pix.item() * real.size(0)
                running_loss_E_rec_fea += loss_E_rec_fea.item() * real.size(0)

                #pbar.set_postfix(**{"Loss D": loss_D.item(), "Loss D adv": loss_D_adv.item(), "Loss D gp": loss_D_r1penalty.item(), "loss E": loss_E.item(), "Loss E adv": loss_E_adv.item(), "Loss E rec pix": loss_E_rec_pix.item(), "Loss E rec fea": loss_E_rec_fea.item()})
                #pbar.update(real.shape[0])

            epoch_loss_D = running_loss_D / len(data_loaders[phase].dataset)
            epoch_loss_D_adv = running_loss_D_adv / len(data_loaders[phase].dataset)
            epoch_loss_D_r1penalty = running_loss_D_r1penalty / len(data_loaders[phase].dataset)

            epoch_loss_E = running_loss_E / len(data_loaders[phase].dataset)
            epoch_loss_E_adv = running_loss_E_adv / len(data_loaders[phase].dataset)
            epoch_loss_E_rec_pix = running_loss_E_rec_pix / len(data_loaders[phase].dataset)
            epoch_loss_E_rec_fea = running_loss_E_rec_fea / len(data_loaders[phase].dataset)

            log_message = f', epoch_loss_D_r1penalty: {epoch_loss_D_r1penalty:.4f}'
            log_message += f', epoch_loss_D_adv: {epoch_loss_D_adv:.4f}'
            log_message += f', epoch_loss_D: {epoch_loss_D:.4f}'

            log_message = f'epoch_loss_E_rec_pix: {epoch_loss_E_rec_pix:.4f}'
            log_message += f', epoch_loss_E_rec_fea: {epoch_loss_E_rec_fea:.4f}'
            log_message += f', epoch_loss_E_adv: {epoch_loss_E_adv:.4f}'
            log_message += f', epoch_loss_E: {epoch_loss_E:.4f}'

            print(f'Phase: {phase}, Epoch: {epoch:05d}, {log_message}')

            util_report_inverter.show_cae_images(general_reports_dir=general_reports_dir, epoch=epoch,  data_loaders=data_loaders, phase=phase, encoder=encoder,  decoder=decoder, device=device, n_img=10)

            # Update history
            # loss D
            history[f'{phase}_loss_disc'].append(epoch_loss_D)
            history[f'{phase}_loss_disc_adv'].append(epoch_loss_D_adv)
            history[f'{phase}_loss_disc_r1penalty'].append(epoch_loss_D_r1penalty)
            # loss E
            history[f'{phase}_loss_enc'].append(epoch_loss_E)
            history[f'{phase}_loss_enc_adv'].append(epoch_loss_E_adv)
            history[f'{phase}_loss_enc_rec_pix'].append(epoch_loss_E_rec_pix)
            history[f'{phase}_loss_enc_rec_fea'].append(epoch_loss_E_rec_fea)

            util_report_inverter.plot_training(history=history, plot_training_dir=plot_training_dir, phase=phase)

            # Deep copy the model
            if phase == 'val_iid':
                if (epoch_loss_E_rec_pix+epoch_loss_E_rec_fea) < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss_E_rec_pix+epoch_loss_E_rec_fea
                    best_encoder_wts = copy.deepcopy(encoder.state_dict())

        util_report_inverter.show_gan_images(general_reports_dir=general_reports_dir, epoch=epoch, noise=fixed_noise, decoder=decoder)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:0f}'.format(best_epoch))
    print('Best val_ood loss: {:4f}'.format(best_loss))

    # load best model weights
    encoder.load_state_dict(best_encoder_wts)
    # Save model
    torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder.pt"))
    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return encoder, history

def get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()

def postprocess(images, min_val=-1.0, max_val=1.0, image_channels=1):
    """Postprocesses the output images if needed.
    This function assumes the input numpy array is with shape [batch_size, channel, height, width]. Here, `channel = 3` for color image and `channel = 1` for grayscale image. The returned images are with shape
    [batch_size, height, width, channel].
    NOTE: The channel order of output images will always be `RGB`.
    Args:
      images: The raw outputs from the generator.
      min_val
      max_val
      image_channels
    Returns:
      The postprocessed images with dtype `numpy.uint8` and range [0, 255].
    Raises:
      ValueError: If the input `images` are not with type `numpy.ndarray` or not with shape [batch_size, channel, height, width].
    """
    if not isinstance(images, np.ndarray):
      raise ValueError(f'Images should be with type `numpy.ndarray`!')

    if images.ndim != 4 or images.shape[1] != image_channels:
        raise ValueError(f'Input should be with shape [batch_size, channel, height, width], where channel equals to {image_channels}!\n But {images.shape} is received!')
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)

    # images = images.transpose(0, 2, 3, 1)

    return images

def batch_norm(z, z_batch, eps=1e-5):
    assert len(z.shape) == 4
    assert len(z_batch.shape) == 4

    mean = z_batch.mean(dim=(0, 2, 3), keepdim=True)
    var = ((z_batch - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

    z_hat = (z - mean) / torch.sqrt(var + eps)
    return z_hat

def particle_pos(source_interim_dir, iid_class,  n_particles=32, dim_space=100):

    data = np.ones((n_particles, dim_space), dtype='float32')
    with open(os.path.join(source_interim_dir, f'particles_position_iic_class_{iid_class}.pkl'), 'rb') as f:
        history_particles = pickle.load(f)
    for particle_idx, p_key in enumerate(history_particles.keys()): # get the last iteration
        data[particle_idx, :] = history_particles[p_key].iloc[-1, :]

    return torch.tensor(data, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

def invert(general_reports_dir, plot_training_dir, x, generator, encoder, device, iterations=500, learning_rate=1e-2, loss_pix_weight=1.0, loss_reg_weight=2.0, tolerance=0.0001, num_vis=10, early_stopping=100000):
    """Inverts the given image to a latent code.
    Basically, this function is based on gradient descent algorithm.
    Args:
      general_reports_dir:
      plot_training_dir:
      x: Target image to invert, which is assumed to have already been  preprocessed.
      generator:
      encoder:
      device:
      iterations:
      learning_rate:
      loss_pix_weight:
      loss_reg_weight:
      tolerance:
      num_vis: Number of intermediate outputs to visualize. (default: 0)
      early_stopping: default 100000
    Returns:
      A three-element tuple. First one is the inverted code. Second one is a list
        of intermediate results, where first image is the input image, second
        one is the reconstructed result from the initial latent code, remainings
        are from the optimization process every `self.iteration // num_viz`
        steps.
    """
    assert x.shape[0] == 1 # only one image at time

    encoder.eval()
    generator.eval()

    x = x.to(device) # image to be inverted
    x.requires_grad = False

    # Initialization: extract z_init using pretrained encoder
    init_z = encoder(x.float())
    z = torch.Tensor(init_z.detach())
    z = z.to(device)
    z.requires_grad = True

    optimizer = torch.optim.Adam([z], lr=learning_rate)

    history_vis = []
    history = util_general.list_dict()
    history_vis.append(postprocess(get_tensor_value(x))[0]) # real image
    x_init_inv = generator(z)
    history_vis.append(postprocess(get_tensor_value(x_init_inv))[0]) # initial reconstructed image

    step = 0
    while step < (iterations + 1):
        running_loss = 0.0

        # Reconstruction loss.
        x_rec = generator(z.float())
        loss_pix = torch.mean((x.float() - x_rec.float()) ** 2)
        running_loss = running_loss + (loss_pix * loss_pix_weight)
        log_message = f'loss_pix: {loss_pix:.3f}'

        # Perceptual loss.
        # todo perceptual loss

        # Regularization loss.
        z_rec = encoder(x_rec.float())
        loss_reg = torch.mean((z.float() - z_rec.float()) ** 2)
        running_loss = running_loss + loss_reg * loss_reg_weight
        log_message += f', loss_reg: {loss_reg:.3f}'

        log_message += f', loss: {running_loss:.3f}'
        #print(f'Step: {step}, lr: {learning_rate}, {log_message}')

        # Do optimization.
        optimizer.zero_grad()
        running_loss.backward()
        optimizer.step()

        # Statistics and history
        history['loss_pix'].append(loss_pix.item())
        history['loss_reg'].append(loss_reg.item())
        history['loss'].append(running_loss.item())

        if num_vis > 0 and step % (iterations // num_vis) == 0:
            util_report_inverter.show_gan_images(general_reports_dir=general_reports_dir, epoch=step, noise=z,  decoder=generator)
            history_vis.append(postprocess(get_tensor_value(x_rec))[0])

        if len(history['loss']) > early_stopping:
            if abs(history['loss'][-1] - history['loss'][-2]) < tolerance:
                break
        step += 1

    #print(f"Exit iterations: {step}")
    log_message += f', exit iteration: {step:.0f}'
    util_report_inverter.plot_training(history=history, plot_training_dir=plot_training_dir)

    # Format history
    history = pd.DataFrame.from_dict(history, orient='index').transpose()

    return get_tensor_value(z), history, history_vis, log_message









