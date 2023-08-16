## Standard libraries
import os


## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from escnn import gspaces
from escnn.nn import FieldType, GeometricTensor, ELU, R2Conv, InnerBatchNorm
from VAES.utils.utils import HypersphericalUniform, VonMisesFisher
import numpy as np
import math
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.euclidean import Euclidean
import geomstats.backend as gs

gs.random.seed(2020)


class Encoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 depth: int,
                 k: int):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - num_landmarks : number of Kendall shape landmarks
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.latent_dim = k + 1

        # Make the simple feedforward net (apply activation after each layer)
        self.layers = []
        self.layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=5, padding=0))
        self.layers.append(nn.LayerNorm([hidden_dim, 64, 64], eps=1e-6))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        for _ in range(depth - 2):
            self.layers.append(ConvNext(hidden_dim, hidden_dim, padding=0))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.layers.append(nn.Conv2d(hidden_dim, self.latent_dim, kernel_size=5, padding=0))
        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        # for layer in self.layers:
        #     print(layer)
        #     print('in:', x.shape)
        #     x = layer(x)
        #     print('out:', x.shape)
        x = self.net(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 hidden_dim: int,
                 depth: int,
                 k: int):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - num_landmarks : Num of Kendall shape landmarks
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()

        self.latent_dim = k
        self.padding = 4

        # Make the simple feedforward net (apply activation after each layer)
        self.layers = []
        self.layers.append(nn.Conv2d(self.latent_dim, hidden_dim, kernel_size=5, padding=4, padding_mode='zeros'))
        self.layers.append(nn.Upsample(scale_factor=2))
        self.layers.append(nn.LeakyReLU())
        for l in range(depth - 2):
            self.layers.append(ConvNext(hidden_dim, hidden_dim, padding=4, padding_mode='replicate'))
            self.layers.append(nn.Upsample(scale_factor=2))
        self.layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=5, padding=4, padding_mode='replicate'))
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):

        x = self.net(x)

        # print('\nDECODER\n')
        # for layer in self.layers:
        #     print(layer)
        #     print('in:', x.shape)
        #     x = layer(x)
        #     print('out:', x.shape)
        return x


class SVAE(pl.LightningModule):

    def __init__(self,
                 kappa_min,
                 kappa_max,
                 input_dim: int,
                 k: int,
                 hidden_dim: int,
                 depth: int,
                 variational: bool,
                 set_spreadloss: bool,
                 lr=0.0001):
        super().__init__()

        self.spatial_size = 68
        self.variational = variational
        self.k = k
        self.set_spreadloss = set_spreadloss
        self.lr = lr
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.register_buffer('kappa_offset', torch.tensor(self.kappa_min))

        self.hypersphere_dim = k - 1

        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        self.lossss = 0.

        # Creating encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dim, depth, k)
        self.decoder = Decoder(input_dim, hidden_dim, depth, k)
        self.decoder_act_fn = nn.Identity()  # TODO: remove!

        # To be used in the loss
        self.mask = torch.nn.Parameter(build_mask(self.spatial_size), requires_grad=False)


    def reparameterize(self, z_mean, kappa):
        q_z = VonMisesFisher(z_mean, kappa)
        p_z = HypersphericalUniform(2 * (self.k - 1) - 1)

        return q_z, p_z

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """

        # Equivariant encoder
        x = x + torch.randn_like(x) * 0.0001  # TODO: weird bug fix... Prevent constant zero signals to be mapped to NaN
        latent = self.encoder(x)

        firstdim = [latent.shape[0]]

        latent_ = torch.squeeze(latent)
        if firstdim == [1]:
            latent_ = torch.unsqueeze(latent_, 0)

        # APPEARANCE (Mean of VMF)
        mean = latent_[:, :-1]

        mean = mean / torch.linalg.norm(mean, dim=-1, keepdim=True)
        kappa = F.softplus(latent_[:, -1:]) + self.kappa_offset

        # SET APPEARANCE DISTRIBUTION
        q_z, p_z = self.reparameterize(mean, kappa)

        if self.variational:
            z = q_z.rsample()
        else:
            z = mean

        x_hat = self.decoder(z[..., None, None])
        x_hat = self.decoder_act_fn(x_hat)

        return x_hat, q_z, p_z

    def random_sample(self, batch_size):
        p_z = HypersphericalUniform(self.hypersphere_dim)
        sampled_latents = p_z.sample(batch_size)[..., None, None].to(self.device)
        # print(sampled_latents.shape)
        return self.decoder(sampled_latents)

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, y = batch  # We do not need the labels

        x_hat, q_z, p_z  = self.forward(x)
        mu = q_z.loc 
        # SPHERE2 = Hypersphere(dim=self.hypersphere_dim)
        # METRIC = SPHERE2.metric
        # distance = METRIC.dist(mu, mu)

        loss = self.mask * F.mse_loss(x_hat, x, reduction="none")

        recon_loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        KLD = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        # Total loss
        if self.set_spreadloss:
            spread_loss = torch.sum(torch.inner(mu, mu))
            return recon_loss, KLD, spread_loss
        else:
            return recon_loss, KLD

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        if self.set_spreadloss:
            loss, kld, spread_loss = self._get_reconstruction_loss(batch)
            self.log('train_loss (recon)', loss)
            self.log('train_loss (kld)', kld)
            self.log('train_loss (recon+kld+spread)', loss + kld+spread_loss)
            self.log('train_loss (spread)', spread_loss)
            return loss + kld + spread_loss
        else:
            loss, kld = self._get_reconstruction_loss(batch)
            self.log('train_loss (recon)', loss)
            self.log('train_loss (kld)', kld)
            self.log('train_loss (recon+kld)', loss + kld)
            return loss + kld 


    def validation_step(self, batch, batch_idx):
        if self.set_spreadloss:
            loss, kld, spread_loss = self._get_reconstruction_loss(batch)
            self.log('val_loss (recon)', loss)
            self.log('val_loss (kld)', kld)
            self.log('val_loss (recon+kld+spread)', loss + kld+spread_loss)
            self.log('val_loss (spread)', spread_loss)
            return loss + kld + spread_loss
        else:
            loss, kld = self._get_reconstruction_loss(batch)
            self.log('val_loss (recon)', loss)
            self.log('val_loss (kld)', kld)
            self.log('val_loss (recon+kld)', loss + kld)
            return loss + kld 

    def test_step(self, batch, batch_idx):
        if self.set_spreadloss:
            loss, kld, spread_loss = self._get_reconstruction_loss(batch)
            self.log('test_loss (recon)', loss)
            self.log('test_loss (kld)', kld)
            self.log('test_loss (recon+kld+spread)', loss + kld+spread_loss)
            self.log('test_loss (spread)', spread_loss)
            return loss + kld + spread_loss
        else:
            loss, kld = self._get_reconstruction_loss(batch)
            self.log('test_loss (recon)', loss)
            self.log('test_loss (kld)', kld)
            self.log('test_loss (recon+kld)', loss + kld)
            return loss + kld 

    def on_train_epoch_end(self):
        # Increase the kappa offset
        if self.variational:
            self.kappa_offset *= math.exp(math.log(self.kappa_max/self.kappa_min) / (2 * self.trainer.max_epochs - 1))
            self.kappa_offset = torch.clip(self.kappa_offset, self.kappa_min, self.kappa_max)
        print('Increased kappa offset to', self.kappa_offset)

    def on_load_checkpoint(self, checkpoint):
        self.train()


# def build_mask(s, margin=2, dtype=torch.float32):

#     mask = torch.ones(64, 64, dtype=torch.float32)
#     mask = F.pad(mask, (2,2,2,2), "constant", 0) 
#     mask = mask[(None,)* 2]
#     return mask
def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask


class ConvNext(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, input_dim, output_dim, padding, padding_mode='replicate'):
        super().__init__()

        self.crop = int((2 * padding - 4)/2)
        self.skip = (input_dim == output_dim)
        self.padding_mode = padding_mode
        hidden_dim = 4 * input_dim # Hidden dim always same!

        self.dwconv = nn.Conv2d(input_dim, input_dim, kernel_size=5, padding=padding, groups=input_dim, padding_mode=padding_mode)  # depthwise conv
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(input_dim, hidden_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.LeakyReLU()
        self.pwconv2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if self.skip:
            if self.crop < 0:
                x = input[:,:,-self.crop:self.crop,-self.crop:self.crop] + x
            elif self.crop > 0:
                x = F.pad(input, (self.crop, self.crop, self.crop, self.crop), self.padding_mode) + x
            else:
                x = input + x
        return x