import torch
import torch.nn.functional as F
from tqdm import tqdm

def generate_samples(images, model, device):
    with torch.no_grad():
        images = images.to(device)
        x_tilde, _, _ = model(images)
    return x_tilde

def train(data_loader, model, optimizer, beta, device):
    print('Training')

    running_loss = 0.0
    running_loss_recons= 0.0
    running_loss_vq= 0.0
    for images, _ in tqdm(data_loader):
        images = images.to(device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + beta * loss_commit
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_loss_recons += loss_recons.item() * images.size(0)
        running_loss_vq += loss.item() * images.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_loss_recons = running_loss_recons / len(data_loader.dataset)
    epoch_loss_vq = running_loss_vq / len(data_loader.dataset)

    return epoch_loss, epoch_loss_recons, epoch_loss_vq

def valid(data_loader, model, device):
    print('Test')
    running_loss_recons= 0.0
    running_vq= 0.0
    with torch.no_grad():
        for images, _ in tqdm(data_loader):
            images = images.to(device)
            x_tilde, z_e_x, z_q_x = model(images)
            running_loss_recons += F.mse_loss(x_tilde, images).item() * images.size(0)
            running_vq += F.mse_loss(z_q_x, z_e_x).item() * images.size(0)

        epoch_loss_recons = running_loss_recons / len(data_loader.dataset)
        epoch_loss_vq = running_vq / len(data_loader.dataset)

    return epoch_loss_recons, epoch_loss_vq
