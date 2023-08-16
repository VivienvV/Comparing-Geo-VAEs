## Standard libraries
import os


## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


import numpy as np
import math
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
# from .types_ import *


class VanillaVAE(pl.LightningModule):


    def __init__(self,
                 original_input_channels: int,
                 latent_dim: int,
                 hidden_dim: int,
                 kernel_size: int,  
                 depth: int, 
                 variational: bool,          
                 lr=0.0001,
                 **kwargs):

        super(VanillaVAE, self).__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.variational = variational
        if variational:
            self.encoder_output_dim =latent_dim + 1 # + 1 for log variance/sigma 
        else:
            self.encoder_output_dim = latent_dim
        self.spatial_size = 68
        self.lr = lr

        # ------------------------------------------------------------
        # --------- ENCODER-------------------------------------------
        # ------------------------------------------------------------   

        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(original_input_channels, hidden_dim,
                            kernel_size=kernel_size, padding  = 0),
                nn.LayerNorm([hidden_dim, 64, 64], eps=1e-6),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            )
        for _ in range(depth - 2):
            modules.append(ConvNext(hidden_dim, hidden_dim, padding=0))
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        modules.append(nn.Conv2d(hidden_dim, self.encoder_output_dim, kernel_size=kernel_size, padding=0))

        self.encoder = nn.Sequential(*modules)

        # ------------------------------------------------------------
        # --------- DECODER-------------------------------------------
        # ------------------------------------------------------------   

        modules = []

        modules.append(nn.Conv2d(latent_dim, hidden_dim, kernel_size=kernel_size, padding=4, padding_mode='zeros'))
        modules.append(nn.Upsample(scale_factor=2))
        modules.append(nn.LeakyReLU())
        for _ in range(depth-2):
            modules.append(ConvNext(hidden_dim, hidden_dim, padding=4, padding_mode='replicate'))
            modules.append(nn.Upsample(scale_factor=2))

        modules.append(nn.Conv2d(hidden_dim, original_input_channels, kernel_size=5, padding=4, padding_mode='replicate'))
        
        modules.append(nn.Identity())

        self.decoder = nn.Sequential(*modules) # Outputs activated output 
        self.mask = torch.nn.Parameter(build_mask(self.spatial_size), requires_grad=False)


    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        return result

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)

        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        latent = self.encode(input)

        firstdim = [latent.shape[0]]

        latent_ = torch.squeeze(latent)

        if firstdim == [1]:
            latent_ = torch.unsqueeze(latent_, 0)

        if self.variational:
            mu = latent_[:, :-1]

            log_var = F.softplus(latent_[:, -1:])

            z = self.reparameterize(mu, log_var)
            z = z[..., None, None]

        else:
            z = latent_[..., None, None]
            mu = latent_
            log_var = None

        return self.decode(z), mu, log_var


    def _get_reconstruction_loss(self, batch):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        x, y = batch  # We do not need the labels

        x_hat, mu, log_var = self.forward(x)

        recon_loss = self.mask * F.mse_loss(x_hat, x, reduction="none")

        recon_loss = recon_loss.sum(dim=[1, 2, 3]).mean(dim=[0])

        if self.variational:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        else:
            kld_loss = None
        return recon_loss, kld_loss

    def random_sample(self,
               num_samples:int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z[..., None, None].to(self.device)
        samples = self.decode(z)
        return samples

    def reconstruct(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss, kld = self._get_reconstruction_loss(batch)
        self.log('train_loss (recon)', loss)
        if self.variational:
            self.log('train_loss (kld)', kld)
            self.log('train_loss (recon+kld)', loss + kld)
            return loss + kld
        else:
            return loss

    def validation_step(self, batch, batch_idx):
        loss, kld = self._get_reconstruction_loss(batch)
        self.log('val_loss (recon)', loss)
        if self.variational:
            self.log('val_loss (kld)', kld)
            self.log('val_loss (recon+kld)', loss + kld)
            return loss + kld
        else:
            return loss

    def test_step(self, batch, batch_idx):
        loss, kld = self._get_reconstruction_loss(batch)
        self.log('test_loss (recon)', loss)
        if self.variational:
            self.log('test_loss (kld)', kld)
            self.log('test_loss (recon+kld)', loss + kld)
            return loss + kld
        else:
            return loss

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


# class convblock(nn.Module):


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