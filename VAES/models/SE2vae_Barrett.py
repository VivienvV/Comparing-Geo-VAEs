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

from VAES.se2cnn.nn import R2ToSE2Conv, SE2ToSE2Conv, SE2ToR2Conv, SE2ToSE2ConvNext, Fourier, SE2LayerNorm, SpatialMaxPool, SE2ToR2Projection, SpatialUpsample

# from .types_ import *


class SE2VAE(pl.LightningModule):


    def __init__(self,
                 original_input_channels: int,
                 latent_dim: int,
                 N: int,
                 hidden_dim: int,
                 kernel_size: int,  
                 depth: int, 
                 variational: bool,          
                 lr=0.0001,
                 **kwargs):

        super(SE2VAE, self).__init__()
        self.save_hyperparameters()

        self.variational = variational
        self.latent_dim = latent_dim
        self.fourier = Fourier(N)
        if variational:
            self.encoder_output_dim =latent_dim + 1 + 1# + 1 for log variance/sigma 
        else:
            self.encoder_output_dim = latent_dim + 1 #plus 1 for theta
        self.spatial_size = 68
        self.lr = lr
        self.N = N


        # ------------------------------------------------------------
        # --------- ENCODER-------------------------------------------
        # ------------------------------------------------------------   

        modules = []

        modules.append(R2ToSE2Conv(original_input_channels, hidden_dim, N, kernel_size=kernel_size, padding=0))
        modules.append(SE2LayerNorm(hidden_dim))
        modules.append(nn.LeakyReLU())
        modules.append(SpatialMaxPool(kernel_size=2, stride=2, padding=0, nbOrientations=N))

        for l in range(depth - 2):
            modules.append(SE2ToSE2ConvNext(hidden_dim, hidden_dim, N, kernel_size=kernel_size, padding=0))
            modules.append(SpatialMaxPool(kernel_size=2, stride=2, padding=0, nbOrientations=N))
            

        modules.append(SE2ToSE2Conv(hidden_dim, self.encoder_output_dim, N, kernel_size=kernel_size, padding=0))
        self.encoder = nn.Sequential(*modules)
        # print(self.encoder)
        # print(self.encoder_output_dim)

        # ------------------------------------------------------------
        # --------- DECODER-------------------------------------------
        # ------------------------------------------------------------   

        modules = []

        modules.append(SE2ToSE2ConvNext(latent_dim, hidden_dim, N, kernel_size=kernel_size, padding=(kernel_size-1)))
        modules.append(SpatialUpsample(2, N))
        modules.append(nn.LeakyReLU())

        for _ in range(depth-2):
            modules.append(SE2ToSE2ConvNext(hidden_dim, hidden_dim, N, kernel_size=kernel_size, padding=(kernel_size-1)))
            modules.append(SpatialUpsample(2, N))

        modules.append(SE2ToR2Conv(hidden_dim, original_input_channels, N, kernel_size=5, padding=(kernel_size-1)))
        modules.append(nn.Identity())

        self.decoder = nn.Sequential(*modules)
        self.mask = torch.nn.Parameter(build_mask(self.spatial_size), requires_grad=False)


    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        return self.encoder(input)

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)

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

        if self.variational:
            z_mu = latent[:, :self.latent_dim]  # [B, k, N, X, Y]
            z_theta = latent[:, self.latent_dim:self.latent_dim+1]  # [B, 1, N, X, Y]
            z_log_var = latent[:, self.latent_dim:self.latent_dim+1]  # [B, 1, N, X, Y]

            mu = self.fourier.band_limit_signal(z_mu)  # [B, k, N, ...]
            log_var = self.fourier.regular_to_irrep(z_log_var, 0)[:, :, 0]  # [B, 1, 2, X, Y] -> [B, 1, X, Y], note only the 1st coordinate is non-zero

            if self.N > 2:
                pose = self.fourier.regular_to_irrep(z_theta, 1)  # [B, 1, 2, X, Y]
                pose_theta = torch.atan2(pose[:,:,1],pose[:,:,0])  # [B, 1, X, Y]
            else:
                pose_theta = None
            if self.N > 2:
                mu_aligned = self.fourier.rotate_signal(mu, -pose_theta)
            else:
                mu_aligned = mu

            gauss_mu = mu_aligned.flatten(1, 2)[..., 0, 0]  # [B, k, 2 or N, X=1, Y=1] -> [B, k*2 or k*N]
            gauss_log_var = F.softplus(log_var)[..., 0, 0]  # [B, 1, X, Y] -> [B, 1]
        
            sampled_landmarks_aligned = self.reparameterize(gauss_mu, gauss_log_var)
            sampled_landmarks_aligned = sampled_landmarks_aligned.unflatten(1, [self.latent_dim, -1])[...,None,None] 
            if self.N > 2:
                sampled_landmarks = self.fourier.rotate_signal(sampled_landmarks_aligned, pose_theta)
            else:
                sampled_landmarks = sampled_landmarks_aligned

            # Equivariant decoder
            z = sampled_landmarks # ([64, 10, 8, 1, 1])

        # autoencoder case    
        else:

            z_mu = latent[:, :self.latent_dim]  # [B, k, N, X, Y]
            z_theta = latent[:, self.latent_dim:self.latent_dim+1]
            
            mu = self.fourier.band_limit_signal(z_mu)  # [B, k, N, ...]
            if self.N > 2:
                pose = self.fourier.regular_to_irrep(z_theta, 1)  # [B, 1, 2, X, Y]
                pose_theta = torch.atan2(pose[:,:,1],pose[:,:,0])  # [B, 1, X, Y]
            else:
                pose_theta = None
            if self.N > 2:
                mu_aligned = self.fourier.rotate_signal(mu, -pose_theta)
            else:
                mu_aligned = mu

            gauss_mu = mu_aligned.flatten(1, 2)[..., 0, 0]  # [B, k, 2 or N, X=1, Y=1] -> [B, k*2 or k*N]

            sampled_landmarks_aligned = gauss_mu.unflatten(1, [self.latent_dim, -1])[...,None,None]  # [B, k * 2 or k * N, 1, 1]

            z = sampled_landmarks_aligned
            gauss_log_var = None

        return  self.decode(z), gauss_mu, gauss_log_var

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
        # print('xhat: {}, mu: {}, var: {}'.format(x_hat.shape, mu.shape, log_var.shape))

        # recon_loss = self.mask * F.mse_loss(x_hat, x, reduction="none")
        recon_loss =  F.mse_loss(x_hat, x, reduction="none")
        

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
        z = torch.randn(num_samples, self.latent_dim*self.N)

        z = z[..., None, None].unflatten(1, [self.latent_dim, self.N]).to(self.device)
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
