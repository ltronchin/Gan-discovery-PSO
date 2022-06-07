import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.optim as optim

from src.hands_on.vq_vae.utils import util_function

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

def get_model(name, channels_img, embedded_dim, num_embedding, data_pso, num_hiddens=64, features_g=64, features_d=64):
    if name == 'vqvae_mnist':
        return VectorQuantizedVAE_MNIST(channels_img, embedded_dim, num_embedding, data_pso, num_hiddens)
    elif name == 'vqvae':
        return VectorQuantizedVAE(channels_img, embedded_dim, num_embedding, data_pso)
    elif name == 'vqvae_dcgan':
        return VectorQuantizedVAE_GAN(channels_img, embedded_dim, num_embedding, data_pso, features_g, features_d)
    else:
        raise ValueError(name)

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


@torch.no_grad()
def pso_weights(model, data):
    data_tensor = torch.tensor(data.values)
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.data = data_tensor


class Generator(nn.Module):
    def __init__(self, embedded_dim, channels_img, features_g):  # z_dim represent the noise dimension
        super(Generator, self).__init__()

        # in the original DCGAN implementation, the batch normalization is not used in the first layer of the discriminator and in the last layer of the generator
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(in_channels=embedded_dim, out_channels=features_g * 2, kernel_size=7, stride=1, padding=0), # 7 x 7
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

class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        print("\nWeights Initialization")
        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


class VQEmbedding(nn.Module):
    def __init__(self, K, D, data_pso):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        if data_pso is not None:
            pso_weights(self.embedding, data_pso)
        else:
            self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = util_function.vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = util_function.vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,  dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlockBatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.block(x)

class VectorQuantizedVAE(nn.Module):
    def __init__(self, channels_img, embedded_dim=64, num_embedding=512, data_pso=None):
        super().__init__()
        dim = embedded_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(channels_img, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlockBatchNorm(dim),
            ResBlockBatchNorm(dim),
        )

        self.codebook = VQEmbedding(num_embedding, embedded_dim, data_pso=data_pso)

        self.decoder = nn.Sequential(
            ResBlockBatchNorm(dim),
            ResBlockBatchNorm(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, channels_img, 4, 2, 1),
            nn.Tanh()
        )

        print("\nWeights Initialization")
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

class VectorQuantizedVAE_MNIST(nn.Module):
    def __init__(self, channels_img, embedded_dim=64, num_embedding=512, data_pso=None, num_hiddens=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels_img, num_hiddens//2, 4, 2, 1), # 32
            nn.ReLU(True),
            nn.Conv2d(num_hiddens//2, num_hiddens, 4, 2, 1), # 64
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, embedded_dim, 7, 2, 0),
        )

        self.codebook = VQEmbedding(K=num_embedding, D=embedded_dim, data_pso=data_pso)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedded_dim, num_hiddens, 7, 2, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens, num_hiddens//2, 4, 2, 1), # 32
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens//2, channels_img, 4, 2, 1), # 1
            nn.Tanh()
        )

        print("\nWeights Initialization")
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)

        return x_tilde, z_e_x, z_q_x



class VectorQuantizedVAE_GAN(nn.Module):
    def __init__(self, channels_img, embedded_dim=64, num_embedding=512,  data_pso=None, features_g=64, features_d=64):
        super().__init__()

        # in the original DCGAN implementation, the batch normalization is not used in the first layer of the discriminator and in the last layer of the generator
        self.encoder = nn.Sequential(
            # Input: N x channels_img, 28, 28
            nn.Conv2d(in_channels=channels_img, out_channels=features_d, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.LeakyReLU(0.2),
            self._block_encoder(in_channels=features_d, out_channels=features_d * 2, kernel_size=4, stride=2, padding=1), # 7x7
            nn.Conv2d(in_channels=features_d * 2, out_channels=embedded_dim, kernel_size=7, stride=2, padding=0),
        )

        self.codebook = VQEmbedding(K=num_embedding, D=embedded_dim, data_pso=data_pso)

        self.decoder = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block_decoder(in_channels=embedded_dim, out_channels=features_g * 2, kernel_size=7, stride=1, padding=0),  # 7 x 7
            self._block_decoder(in_channels=features_g * 2, out_channels=features_g, kernel_size=4, stride=2, padding=1),  # 14 x 14
            nn.ConvTranspose2d(in_channels=features_g, out_channels=channels_img, kernel_size=4, stride=2, padding=1), # 28 x 28
            nn.Tanh(), # [-1, 1] if the real images are rescaled in [-1 1] also the synthetic images must be rescaled in the same range
        )

        print("\nWeights Initialization")
        self.apply(weights_init)

    def _block_encoder(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def _block_decoder(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # we neeed to upsample
        )

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        print("\nWeights Initialization")
        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x