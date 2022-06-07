import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import itertools

import torch
from torch import nn


def get_training_loop(task,  encoder,  decoder,  device, dataloader,  loss_fn, optimizer, noise_factor):
    if task == 'reconstruction':
        return rec_ae_train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer)
    elif task == 'denoising':
        return den_ae_train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor)
    else:
        raise ValueError(task)

def get_test_loop(task, general_reports_dir, encoder, decoder, device, dataloader, dataset, loss_fn, noise_factor):
    if task == 'reconstruction':
        return rec_ae_test_epoch(general_reports_dir, encoder, decoder, device, dataloader, dataset, loss_fn)
    elif task == 'denoising':
        return den_ae_test_epoch(general_reports_dir, encoder, decoder, device, dataloader, dataset, loss_fn, noise_factor)
    else:
        raise ValueError(task)

def add_noise(inputs, noise_factor=0.3):
    noise = inputs + torch.randn_like(inputs) * noise_factor
    noise = torch.clip(noise, 0., 1.)
    return noise

def load_autoencoder(model_dir, latent_space, device):  # Pretrained CAE encoder
    encoder = Encoder(encoded_space_dim=latent_space)
    decoder = Decoder(encoded_space_dim=latent_space)

    encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pt"),  map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(model_dir,  "decoder.pt"),  map_location=device))

    encoder.to(device)
    decoder.to(device)
    return encoder, decoder

def create_encoded_sample_batch(encoder, dataloader, batch_size, device, n_iter=4):

    samples = []
    encoded_samples = []
    # Loop
    encoder.eval()
    with torch.no_grad():
        # with tqdm(total = n_iter * batch_size, unit='img') as pbar:
        for i, (x_batch, y_batch) in enumerate(dataloader):
            assert x_batch[0].dtype == torch.float32
            assert torch.max(x_batch[0]) <= 1.0
            assert torch.min(x_batch[0]) >= 0.0
            assert x_batch[0].shape[0] == 1
            if i >= n_iter:
                break

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            samples.append(x_batch.detach().cpu().numpy())
            # Prediction
            encoded_batch = encoder(x_batch.float())

            for encoded_img, label in zip(encoded_batch, y_batch):
                encoded_img = encoded_img.flatten().cpu().numpy()
                encoded_sample = {f"var_{i}": enc for i, enc in enumerate(encoded_img)}
                encoded_sample['label'] = label.detach().cpu().numpy()
                encoded_samples.append(encoded_sample)
            # pbar.update(x_batch.shape[0])
        encoded_samples = pd.DataFrame(encoded_samples) # convert from list to dataframe
        samples = np.asarray(list(itertools.chain(*samples)))
        if np.isnan(encoded_samples['label'][0]):
            encoded_samples = encoded_samples.drop(['label'], axis=1)

    return encoded_samples, samples

def create_encoded_sample(encoder, data, device):
    encoded_samples = []
    for sample in data:
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        # Encode image
        encoder.eval()
        with torch.no_grad():
            encoded_img  = encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"var_{i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples) # convert from list to dataframe
    return encoded_samples

def encode_features():
    pass

def decode_features():
    pass


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        # Encoder architecture (input 1x28x28)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU()
        )
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear section
        self.encoder_linear = nn.Sequential(
            nn.Linear(in_features=3 * 3 * 32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=encoded_space_dim)
        )

    def forward(self, x):  # Define the forward pass
        x = self.encoder_cnn(x)  # Apply convolutions
        x = self.flatten(x)  # Flatten
        x = self.encoder_linear(x)  # Apply linear layers
        return x

class Decoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        # Linear section
        self.decoder_linear = nn.Sequential(
            nn.Linear(in_features=encoded_space_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=3 * 3 * 32),
            nn.ReLU()
        )

        # Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        # Convolutional section
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, output_padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_linear(x)  # Apply linear layers
        x = self.unflatten(x)  # Unflatten
        x = self.decoder_conv(x)  # Apply transposed convolutions
        x = torch.sigmoid(x)  # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        return x

# Training function
def den_ae_train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor=0.3):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()

    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        image_noisy = add_noise(image_batch, noise_factor)
        image_noisy = image_noisy.to(device)  # Move tensor to the proper device
        image_batch = image_batch.to(device) # Move tensor to the proper device

        encoded_data = encoder(image_noisy)  # Encode data
        decoded_data = decoder(encoded_data)  # Decode data

        loss = loss_fn(decoded_data, image_batch)  # Evaluate loss
        loss.backward()  # Backward pass (Computes the gradient)
        optimizer.step()  # Update the weights

        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss), encoder, decoder

# Testing function
def den_ae_test_epoch(general_reports_dir, encoder, decoder, device, dataloader, dataset, loss_fn, noise_factor=0.3):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()

    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch, noise_factor)
            image_noisy = image_noisy.to(device)

            encoded_data = encoder(image_noisy)  # Encode data
            decoded_data = decoder(encoded_data)  # Decode data

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        val_loss = loss_fn(conc_out, conc_label)  # Evaluate global loss

        plot_den_ae_outputs(
            val_dataset=dataset, encoder=encoder, decoder=decoder, device=device, general_reports_dir=general_reports_dir, noise_factor=noise_factor
        )

    return val_loss.data, encoder, decoder

# Training function
def rec_ae_train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()

    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        image_batch = image_batch.to(device) # Move tensor to the proper device

        encoded_data = encoder(image_batch)  # Encode data
        decoded_data = decoder(encoded_data)  # Decode data

        loss = loss_fn(decoded_data, image_batch)  # Evaluate loss
        loss.backward()  # Backward pass (Computes the gradient)
        optimizer.step()  # Update the weights

        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss), encoder, decoder

# Testing function
def rec_ae_test_epoch(general_reports_dir, encoder, decoder, device, dataloader, dataset, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()

    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)

            encoded_data = encoder(image_batch)  # Encode data
            decoded_data = decoder(encoded_data)  # Decode data

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        val_loss = loss_fn(conc_out, conc_label)  # Evaluate global loss

        plot_rec_ae_outputs(
            val_dataset=dataset, encoder=encoder, decoder=decoder, device=device, general_reports_dir=general_reports_dir)

    return val_loss.data, encoder, decoder


def plot_den_ae_outputs(val_dataset, encoder, decoder, device, general_reports_dir, n_img=10, noise_factor=0.3):
    plt.figure(figsize=(9,2))
    for i in range(n_img):

        ax = plt.subplot(3, n_img, i + 1)
        img = val_dataset[i][0].unsqueeze(0)
        image_noisy = add_noise(img, noise_factor)
        image_noisy = image_noisy.to(device)

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            rec_img = decoder(encoder(image_noisy))

        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n_img, i + 1 + n_img)
        plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n_img, i + 1 + n_img + n_img)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False), ax.get_yaxis().set_visible(False)

    plt.savefig(os.path.join(general_reports_dir, "img_loss.png"), dpi=400, format='png')
    plt.show()

def plot_rec_ae_outputs(val_dataset, encoder, decoder, device, general_reports_dir, n_img=10):
    plt.figure(figsize=(9, 2))
    for i in range(n_img):
        img = val_dataset[i][0].unsqueeze(0)
        img = img.to(device)
        encoder.eval()
        decoder.eval()
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

    plt.savefig(os.path.join(general_reports_dir, "img_loss.png"), dpi=400, format='png')
    plt.show()

def plot_img(samples, encoded_synthetic, general_reports_dir, r0, r1, n=10, w=28):
    plt.figure()
    mask_var_1 = ((encoded_synthetic[:, 0] <= r0[1]) * (encoded_synthetic[:, 0] >= r0[0]))
    mask_var_2 = ((encoded_synthetic[:, 1] <= r1[1]) * (encoded_synthetic[:, 1] >= r1[0]))
    mask = mask_var_1 * mask_var_2
    samples = samples[mask]

    img = np.zeros((n * w, n * w))
    for i in range(n):
        for j in range(n):
            x  = samples[np.random.randint(0, len(samples))].reshape(28, 28)
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x

    plt.imshow(img, cmap='gist_gray')
    plt.axis('off')
    plt.savefig(os.path.join(general_reports_dir, f"synthetic_imgs_r0_{r0[0]}_{r0[1]}__r1_{r1[0]}_{r1[1]}.png"), dpi=1200, format='png')

def plot_img_latent_space(decoder, device, general_reports_dir, r0=None, r1=None, n=10, w=28):

    # w img_dim,  n number of row and column
    if r0 is None:
        r0 = (-1, 1)  # x --> first latent feature interval to plot
    if r1 is None:
        r1 = (-1, 1)  # y --> second latent feature interval to plot
    #np.linspace(*r1, n)  # Return evenly spaced numbers over a specified interval.

    plt.figure()
    img = np.zeros((n * w, n * w))  # create superimage
    for i, y in enumerate(np.linspace(*r1, n)):  # row --> draw image per row, starting from the bottom row and the first column
        for j, x in enumerate(np.linspace(*r0, n)):  # column
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()  # Returns a new Tensor, detached from the current graph.
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1], cmap='gist_gray')
    plt.savefig(os.path.join(general_reports_dir, f"img_latent_r0_{r0[0]}_{r0[1]}__r1_{r1[0]}_{r1[1]}.png"), dpi=400, format='png')

def plot_feature_latent_space(general_reports_dir, encoded_samples, dataset='Training'):

    if not isinstance(encoded_samples, pd.DataFrame):
        encoded_samples = pd.DataFrame(encoded_samples)

    assert encoded_samples.shape[1] <= 3

    if 'umap_0' in encoded_samples.columns:
        var_0, var_1 = 'umap_0', 'umap_1'
        scatter_marker = '.'
        fig_title = f'UMAP Latent space {dataset} Set'
        fig_savename = f'umap_latent_space_{dataset}'
    else:
        var_0, var_1 = 'var_0', 'var_1'
        scatter_marker = 'o'
        fig_title = f'Latent space {dataset} Set'
        fig_savename = f'latent_space_{dataset}'

    if 'label' in encoded_samples.columns:
        # Plot training latent space with label
        target = np.unique(encoded_samples["label"])
        for lab in target:
            plt.scatter(encoded_samples[var_0][encoded_samples["label"] == lab],   encoded_samples[var_1][encoded_samples["label"] == lab],
                        label=lab, alpha=1, s=10, marker=scatter_marker, edgecolors='none')
        plt.legend()
    else:
        # Plot synthetic latent space without label
        plt.figure()
        plt.scatter(encoded_samples[var_0], encoded_samples[var_1], alpha=1, s=10, marker=scatter_marker, edgecolors='none')

    plt.xlabel(var_0)
    plt.ylabel(var_1)
    plt.title(fig_title)
    plt.savefig(os.path.join(general_reports_dir, f"{fig_savename}.png"), dpi=400, format='png')
    plt.show()

def plot_feature_latent_space_interactive(encoded_samples, dataset):
    assert encoded_samples.shape[1] <= 3

    try:
        if 'umap_0' in encoded_samples.columns:
            var_0, var_1 = 'umap_0', 'umap_1'
            fig_title = f'UMAP Latent space {dataset} Set'
        else:
            var_0, var_1 = 'var_0', 'var_1'
            fig_title = f'Latent space {dataset} Set'

        fig = px.scatter(encoded_samples, x=var_0, y=var_1, color=encoded_samples.label.astype(str), opacity=0.7, title=fig_title)
        fig.show()
    except:
        print("No label!!!")
