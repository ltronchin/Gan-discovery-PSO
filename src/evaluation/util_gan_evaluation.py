import torch
from torch.distributions import MultivariateNormal

import os
import pandas as pd
import numpy as np
import scipy
import seaborn as sns # This is for visualization
import matplotlib.pyplot as plt

from src.utils import util_general

#############################################################
# Frechet Inception Distance function
#############################################################
def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

def matrix_sqrt(x):
    """
    Function that takes in a matrix and returns the square root of that matrix.
    For an input matrix A, the output matrix B would be such that B @ B is the matrix A.
    Parameters:
        x: a matrix
    """
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)

def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    """
    Function for returning the Fréchet distance between multivariate Gaussians,
    parameterized by their means and covariance matrices.
    Parameters:
        mu_x: the mean of the first Gaussian, (n_features)
        mu_y: the mean of the second Gaussian, (n_features)
        sigma_x: the covariance matrix of the first Gaussian, (n_features, n_features)
        sigma_y: the covariance matrix of the second Gaussian, (n_features, n_features)
    """
    fid = torch.sum((mu_x - mu_y)**2) + torch.trace(sigma_x + sigma_y - 2*matrix_sqrt(torch.matmul(sigma_x, sigma_y)))
    return fid

def compute_statistics(encoded_real, encoded_synthetic):
    encoded_real = torch.tensor(encoded_real.iloc[:, :-1].values, dtype=torch.float32)
    encoded_synthetic = torch.tensor(encoded_synthetic.values, dtype=torch.float32)

    mu_real = torch.mean(encoded_real, axis=0)
    covariance_real = get_covariance(encoded_real)
    mu_synthetic = torch.mean(encoded_synthetic, axis=0)  # torch.Size([h_latent_space])
    covariance_synthetic = get_covariance(encoded_synthetic)  # torch.Size([h_latent_space, h_latent_space])

    return mu_real, mu_synthetic, covariance_real, covariance_synthetic

def approximate_multivariate_normal(mean, covariance):
    assert mean.dtype == torch.float32
    assert covariance.dtype == torch.float32
    independent_dist = MultivariateNormal(mean, covariance)
    return independent_dist

def plot_multivariate_normal(mu_real, mu_synthetic, covariance_real, covariance_synthetic, indices):
    if len(indices)<=2:
        distribution_1 = approximate_multivariate_normal(mean=mu_real, covariance=covariance_real)
        distribution_2 = approximate_multivariate_normal(mean=mu_synthetic, covariance=covariance_synthetic)
        samples_1 = distribution_1.sample((5000,))
        samples_2 = distribution_2.sample((5000,))
        sns.jointplot(samples_1[:, 0], samples_1[:, 1], kind="kde")
        sns.jointplot(samples_2[:, 0], samples_2[:, 1], kind="kde")
    else:
        distribution_1 = approximate_multivariate_normal(mean= mu_real[indices], covariance=covariance_real[indices][:, indices])
        distribution_2 = approximate_multivariate_normal(mean= mu_synthetic[indices], covariance=covariance_synthetic[indices][:, indices])
        samples_1 = distribution_1.sample((5000,))
        samples_2 = distribution_2.sample((5000,))
        df_1 = pd.DataFrame(samples_1.numpy(), columns=indices)
        df_2 = pd.DataFrame(samples_2.numpy(), columns=indices)
        df_1["is_real"] = "yes"
        df_2["is_real"] = "no"
        df = pd.concat([df_2, df_1])
        sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')
    plt.show()

#############################################################
# Inception Score function
#############################################################
def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(np.mean(p_yx, axis=0), axis=0) # along images>one mean value for each class given N images
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = np.sum(kl_d, axis=1) # along classes>one value for each image given N_pat class
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score

#############################################################
# Reconstruction loss
#############################################################

def add_noise(inputs, noise_factor=0.3):
    noise = inputs + torch.randn_like(inputs) * noise_factor
    noise = torch.clip(noise, 0., 1.)
    return noise

def test_epoch(encoder, decoder, device, data, loss_fn, noise_factor=0.3):
    # Set evaluation mode for encoder and decoder

    encoder.eval()
    decoder.eval()

    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image in data: # for image in tqdm(data):
            image = torch.tensor(image).unsqueeze(0) # convert to tensor and add a dimension
            image_noisy = add_noise(image, noise_factor)
            image_noisy = image_noisy.to(device)

            encoded_data = encoder(image_noisy)  # Encode data
            decoded_data = decoder(encoded_data)  # Decode data

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image.cpu())

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        val_loss = loss_fn(conc_out, conc_label)  # Evaluate global loss
    return val_loss.data, encoder, decoder

#############################################################
# Polarization across patients
#############################################################

def plot_posterior_polarization(epoch, general_reports_dir, p_yx):

    print("Class polarization")
    iid_class = util_general.iid_class()
    plt.figure()
    clf_mean = np.mean(p_yx.to_numpy(), axis=0)
    clf_index = np.argsort(clf_mean)

    x = np.arange(clf_mean.shape[0])
    y = clf_mean[clf_index]

    plt.xticks(np.arange(len(x)), np.array([iid_class.idx_to_idx_iid_class(clf_idx) for clf_idx in clf_index]))
    plt.plot(x, y)
    plt.xlabel("Classifier/Class")
    plt.ylabel("Medium activation across samples")
    plt.savefig(os.path.join(general_reports_dir, f"class_polarization_{epoch}.png"), dpi=400, format='png')
    plt.show()

#############################################################
# Variance and energy
#############################################################

def getEnergy(posterior):
    return np.sum(posterior ** 2, axis=1)

def getVariance(posterior):
    return np.var(posterior, axis=1)

def plot_histogram(epoch, general_reports_dir, p_yx, variables=None):
    if variables is None:
        variables = ['energy', 'variance']
    for var in variables:
        if var == 'energy':
            bin_width = 0.1
            var_pyx =  getEnergy(p_yx)
        elif var == 'variance':
            var_pyx = getVariance(p_yx)
            bin_width = 0.01
        else:
            raise ValueError(var)
        plt.figure()
        bin_range = np.abs(np.min(var_pyx)) + np.abs(np.max(var_pyx))
        plt.hist(var_pyx, bins=int(bin_range / bin_width), color='blue')
        plt.ylabel("Occurrence")
        plt.xlabel(var)
        plt.savefig(os.path.join(general_reports_dir, f"hist_{var}_{epoch}.png"), dpi=400, format='png')
        plt.show()

        plt.figure()
        sns.histplot(var_pyx, kde=True, stat='density', bins=int(bin_range / bin_width), common_norm=True,
                     color='darkblue', edgecolor=None, palette='Set1', line_kws={"lw": 3})
        plt.xlabel("Variance")
        plt.savefig(os.path.join(general_reports_dir, f"kde_{var}_{epoch}.png"), dpi=400, format='png')
        plt.show()
