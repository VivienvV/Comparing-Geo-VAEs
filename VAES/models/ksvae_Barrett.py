## Standard libraries
import math

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from functools import partial
# import skimage.measure
from VAES.utils.utils import HypersphericalUniform, VonMisesFisher

## For the group convolutions
from VAES.se2cnn.nn import R2ToSE2Conv, SE2ToSE2Conv, SE2ToR2Conv, SE2ToSE2ConvNext, Fourier, SE2LayerNorm, SpatialMaxPool, SE2ToR2Projection, SpatialUpsample


class Encoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 depth: int,
                 k: int,
                 N: int,
                 kernel_size: int):
        """
        """
        super().__init__()

        # Make the simple feedforward net (apply activation after each layer)
        self.layers = []
        # Transition: Input to hidden layer (and lift to SE2)
        self.layers.append(R2ToSE2Conv(input_dim, hidden_dim, N, kernel_size=kernel_size, padding=0))
        self.layers.append(SE2LayerNorm(hidden_dim))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(SpatialMaxPool(kernel_size=2, stride=2, padding=0, nbOrientations=N))

        for l in range(depth - 2):
            self.layers.append(SE2ToSE2ConvNext(hidden_dim, hidden_dim, N, kernel_size=kernel_size, padding=0))
            self.layers.append(SpatialMaxPool(kernel_size=2, stride=2, padding=0, nbOrientations=N))
            
        # Transition: hidden to output layer
        self.layers.append(SE2ToSE2Conv(hidden_dim, k + 1 + 1, N, kernel_size=kernel_size, padding=0)) # out should be torch.Size([128, 7, 8, 1, 1])
        # Turn into sequential neural network
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
                 k: int,
                 N: int,
                 kernel_size: int):
        """
        """
        super().__init__()

        padding = kernel_size - 1
        
        # Make the simple feedforward net (apply activation after each layer)
        self.layers = []
        # Transition: latent to hidden layer (group conv on SE2)
        self.layers.append(SE2ToSE2ConvNext(k, hidden_dim, N, kernel_size=kernel_size, padding=padding))
        self.layers.append(SpatialUpsample(2, N))
        self.layers.append(nn.LeakyReLU())
        # ConvNext layers
        # for l in range(depth - 2):
        for l in range(depth-2):
            self.layers.append(SE2ToSE2ConvNext(hidden_dim, hidden_dim, N, kernel_size=kernel_size, padding=padding))
            self.layers.append(SpatialUpsample(2, N))
        # Transition: hidden to output layer
        # self.layers.append(nn.LeakyReLU())
        self.layers.append(SE2ToR2Conv(hidden_dim, output_dim, N, kernel_size=5, padding=padding))
        # Turn into sequential neural network
        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        # print('\nDECODER\n')
        # for layer in self.layers:
        #     print(layer)
        #     print('in:', x.shape)
        #     x = layer(x)
        #     print('out:', x.shape)
        x = self.net(x)
        return x


class VAES(pl.LightningModule):

    def __init__(self,
                 kappa_min,
                 kappa_max,
                 input_dim: int,
                 k: int,
                 N: int,
                 hidden_dim: int,
                 kernel_size: int,
                 depth: int,
                 variational: bool,
                 set_spreadloss: bool,
                 lr=0.0001,
                 type="regular"):
        super().__init__()
        self.save_hyperparameters()

        # self.spatial_size = depth * (kernel_size - 1) + 1
        self.spatial_size = 68
        self.variational = variational
        self.k = k
        self.set_spreadloss = set_spreadloss
        
        self.N = N
        self.lr = lr
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.register_buffer('kappa_offset', torch.tensor(self.kappa_min))
        self.type = type
        if self.type == "regular":
            self.hypersphere_dim = N * k - 1
        else:
            self.hypersphere_dim = 2 * k - 1

        # Creating encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dim, depth, k, N, kernel_size)
        self.decoder = Decoder(input_dim, hidden_dim, depth, k, N, kernel_size)
        self.fourier = Fourier(N)
        self.decoder_act_fn = nn.Identity()
        # To be used in the loss
        self.register_buffer("mask", build_mask(self.spatial_size))
        self.mask = torch.nn.Parameter(build_mask(self.spatial_size), requires_grad=False)

    def reparameterize(self, mean, kappa):
        q_z = VonMisesFisher(mean, kappa)
        p_z = HypersphericalUniform(self.hypersphere_dim)

        return q_z, p_z

    def random_sample(self, batch_size):
        p_z = HypersphericalUniform(self.hypersphere_dim)
        sampled_latents = p_z.sample(batch_size)[..., None, None].unflatten(1, [self.k, self.N]).to(self.device)
        return self.decoder(sampled_latents)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """

        # Encoder
        x = x + torch.randn_like(x) * 0.0001  # TODO: weird bug fix... Prevent constant zero signals to be mapped to NaN
        z = self.encoder(x)

        # Unpack latent to VMF parametrization
        vmf_mu, vmf_kappa, pose_theta = self.shape_latent_to_vmf(z)
        q_z, p_z = self.reparameterize(vmf_mu, vmf_kappa)
        # sample to landmarks
        if self.variational:
            
            sampled_landmarks_aligned = q_z.rsample().unflatten(1, [self.k, -1])[...,None,None]  # [B, k * 2 or k * N, 1, 1]
        else:
            sampled_landmarks_aligned = vmf_mu.unflatten(1, [self.k, -1])[...,None,None]  # [B, k * 2 or k * N, 1, 1]

        if self.N > 2:
            if self.type == "regular":
                sampled_landmarks = self.fourier.rotate_signal(sampled_landmarks_aligned, pose_theta)
            else:
                sampled_landmarks = self.fourier.rotate_irrep(sampled_landmarks_aligned, pose_theta)
        else:
            sampled_landmarks = sampled_landmarks_aligned

        # Equivariant decoder

        z_sampled = sampled_landmarks
        x_hat = self.decoder(z_sampled)
        x_hat = self.decoder_act_fn(x_hat)

        return x_hat, q_z, p_z

    def shape_latent_to_vmf(self, z):
        z_landmarks = z[:, :self.k]  # [B, k, N, X, Y]
        z_pose = z[:, self.k:self.k+1]  # [B, 1, N, X, Y]
        z_kappa = z[:, self.k + 1: self.k + 2]  # [B, 1, N, X, Y]

        # Convert regular representations (signals) to corresponding types
        if self.type == "regular":  # todo: elif irrep
            # Bandlimit means inverse fourier transform
            landmarks = self.fourier.band_limit_signal(z_landmarks)  # [B, k, N, ...]
        else:
            landmarks = self.fourier.regular_to_irrep(z_landmarks, 1)  # [B, k, 2, ...]
        if self.N > 2:
            pose = self.fourier.regular_to_irrep(z_pose, 1)  # [B, 1, 2, X, Y]
            pose_theta = torch.atan2(pose[:,:,1],pose[:,:,0])  # [B, 1, X, Y]
        else:
            pose_theta = None
        kappa = self.fourier.regular_to_irrep(z_kappa, 0)[:, :, 0]  # [B, 1, 2, X, Y] -> [B, 1, X, Y], note only the 1st coordinate is non-zero

        if self.N > 2:
            if self.type == "regular":
                landmarks_aligned = self.fourier.rotate_signal(landmarks, -pose_theta)
            else:
                landmarks_aligned = self.fourier.rotate_irrep(landmarks, -pose_theta)
        else:
            landmarks_aligned = landmarks
        vmf_mu = landmarks_aligned.flatten(1, 2)[..., 0, 0]  # [B, k, 2 or N, X=1, Y=1] -> [B, k*2 or k*N]
        vmf_mu = vmf_mu / torch.linalg.norm(vmf_mu, dim=1, keepdim=True) # HIERMEE NOG EXPERIMTNEREn
        vmf_kappa = F.softplus(kappa)[..., 0, 0] + self.kappa_offset  # [B, 1, X, Y] -> [B, 1]

        return vmf_mu, vmf_kappa, pose_theta

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, y = batch  # We do not need the labels

        x_hat, q_z, p_z = self.forward(x)
        mu = q_z.loc 
        # Encoder equivariance loss
        loss = self.mask * F.mse_loss(x_hat, x, reduction="none")
        recon_loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        KLD = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        
        if self.set_spreadloss:
            spread_loss = torch.sum(torch.inner(mu, mu))
            # Total loss
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
        self.kappa_offset *= math.exp(math.log(self.kappa_max/self.kappa_min) / (0.5 * self.trainer.max_epochs - 1))
        self.kappa_offset = torch.clip(self.kappa_offset, self.kappa_min, self.kappa_max)
        print('Increased kappa offset to', self.kappa_offset)

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