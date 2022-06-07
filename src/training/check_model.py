import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inverter.utils_vq_vae import util_model_v1

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim,out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(dim),
        )
    def forward(self, x):
        return x + self.block(x)

if __name__ == '__main__':
    channel_img = 1
    image_size = 28
    num_hiddens = 128
    num_residual_layers = 2
    num_residual_hiddens = 32
    embedding_space = 64
    num_embedding_dim = 512

    print("\nSelect device")
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    num_workers = 0 if device.type == "cpu" else 32
    print(f'device: {device}')

    conv = nn.Conv2d(in_channels=channel_img, out_channels=num_hiddens//4, kernel_size=4, stride=2, padding=1)
    conv1 = nn.Conv2d(in_channels=num_hiddens//4, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1)
    conv2 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens, kernel_size=4, stride=2, padding=1)
    conv3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1)
    res_conv = ResBlock(dim=num_hiddens)

    x = torch.randn((1, channel_img, image_size, image_size)).to(device)

    # Enconding
    print('Encoding')
    x = conv(x)
    print(x.shape)
    x = conv1(x)
    print(x.shape)
    x = conv2(x)
    print(x.shape)
    x = conv3(x)
    print(x.shape)
    print('Res conv')
    x = res_conv(x)
    print(x.shape)
    enc_x = res_conv(x)
    print(enc_x.shape)

    assert enc_x.shape == (1, num_hiddens, 1, 1)

    # Decoding
    print('Decoding')
    conv_dec = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_space, kernel_size=3, stride=1, padding=1)

    conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens // 2, kernel_size=3, stride=2)
    conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=num_hiddens // 4, kernel_size=4, stride=2, padding=1, output_padding=1)
    conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens // 4, out_channels=num_hiddens // 4, kernel_size=4,  stride=2, padding=1)
    conv_trans_4 = nn.ConvTranspose2d(in_channels=num_hiddens // 4, out_channels=1, kernel_size=4, stride=2, padding=1)

    x = conv_dec(enc_x)
    print(x.shape)
    x = res_conv(x)
    print(x.shape)
    print("Transposing start")
    x = conv_trans_1(x)
    x = F.relu(x)
    print(x.shape)
    x = conv_trans_2(x)
    x = F.relu(x)
    print(x.shape)
    x = conv_trans_3(x)
    print(x.shape)
    x = conv_trans_4(x)
    print(x.shape)

    assert x.shape == (1, channel_img, image_size, image_size)

    x = torch.randn((1, channel_img, image_size, image_size)).to(device)
    enc = util_model_v1.Encoder(channel_img=channel_img, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)
    pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_space, kernel_size=1, stride=1)
    vq_e = util_model_v1.VectorQuantizer(num_embeddings=num_embedding_dim, embedding_dim=embedding_space, beta=0.25)
    dec = util_model_v1.Decoder(embedding_dim=embedding_space, num_hiddens=num_hiddens,  num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)

    e_x = enc(torch.randn((1, channel_img, image_size, image_size)).to(device))
    assert e_x.shape == (1, num_hiddens, 1, 1)
    e_x = pre_vq_conv(e_x)
    assert e_x.shape == (1, embedding_space, 1, 1)
    e_x_vq = vq_e(e_x)
    assert e_x_vq[0].shape == (1, embedding_space, 1, 1)
    rec_x = dec(e_x_vq[0])
    assert rec_x.shape == (1, 1, image_size, image_size)

    print('Done!')