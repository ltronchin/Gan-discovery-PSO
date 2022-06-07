import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def rescale_torch(img, min_val=None, max_val=None):
    if not min_val:
        min_val = torch.min(img)
    if not max_val:
        max_val = torch.max(img)
    img = (img - min_val) / (max_val - min_val)
    return img

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

# Parameters
latent_dim = 10 # number of features of eache particle
worker = "cuda:0"
gpu_num_workers = 32

channel = 1

units_gen = 64
gan_dir = '/Users/ltronchin/Desktop/Cosbi/Computer Vision/Gan discovery PSO/models/mnist/00006--dcgan.py/' # path to gan

# Device
print("Select device")
device = torch.device(worker if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else gpu_num_workers
print(f'device: {device}')

# Upload Generator
print("Upload gan generator")
checkpoint_g = torch.load(os.path.join(gan_dir, 'best_g.tar'), map_location=torch.device(device))
generator = Generator(z_dim=latent_dim, channels_img=channel, features_g=units_gen)
generator.load_state_dict(checkpoint_g['model_state_dict'])
generator = generator.to(device)

# Prediction

z=torch.randn((1, latent_dim, 1, 1))
print(z.shape)

# Check z latent vector
if z.ndim != 4:
    # print(f"Unsupported input dimension: Expected 4D (batched) input but got input of size: {pos.shape}! Let's add dimension!")
    for _ in range(4 - z.ndim):
        z = torch.unsqueeze(z, dim=0)
    z = z.view([1, latent_dim, 1, 1])

if z.dtype == torch.float64:
    # print( f"Unsupported input type: Expected float32 but got {pos.dtype}! Let's change Tensor dtype!")
    z = z.to(torch.float32)

generator.eval()
with torch.no_grad():
    img = generator(z)

img_rescaled = rescale_torch(img)
img_numpy =   img_rescaled.detach().numpy()[0][0]
plt.imshow(img_numpy, cmap='gray')
plt.show()



