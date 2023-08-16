## Standard libraries
import os


## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy
from escnn import gspaces
from escnn.nn import FieldType, GeometricTensor, ELU, R2Conv, InnerBatchNorm
from VAES.utils.utils import HypersphericalUniform, VonMisesFisher
import numpy as np
import math



class MLP(pl.LightningModule):
  
    def __init__(self,
                 ksvae_model,
                 weight_norm,
                 hidden_dim: int,
                 num_classes: int,
                 lr = 0.0005 
                 ):
        super().__init__()

        self.lr = lr
        self.num_classes = num_classes
        self.save_hyperparameters()
        
        self.ksvae_model = ksvae_model
        self.ksvae_model.eval()

        if self.ksvae_model.__class__.__name__ == 'VanillaVAE' or self.ksvae_model.__class__.__name__ == 'SE2VAE':
            self.k = ksvae_model.latent_dim

        else:
            self.k = ksvae_model.k

        if self.ksvae_model.__class__.__name__ == 'VAES' or self.ksvae_model.__class__.__name__ == 'SE2VAE':            
            self.hypersphere_dim = ksvae_model.N * self.k - 1
            self.input_dim = self.k * ksvae_model.N
        else:           
            self.hypersphere_dim = self.k - 1
            self.input_dim = self.k 
        
        self.accuracy = Accuracy()
        
        self.layers = []

        self.layers.append(nn.Flatten())
        if weight_norm:
            self.layers.append(nn.utils.weight_norm(nn.Linear(self.input_dim, hidden_dim)))
        else:
            self.layers.append(nn.Linear(self.input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, int(hidden_dim/2)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(int(hidden_dim/2), num_classes))
        self.classifier = nn.Sequential(*self.layers)
        self.ce = nn.CrossEntropyLoss()

        # eerste lin layer gewichten normaliseren



    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        
        
        if self.ksvae_model.__class__.__name__ == 'VanillaVAE' or self.ksvae_model.__class__.__name__ == 'SE2VAE':

            xhat, mu, sig = self.ksvae_model(x)            
        else:
            xhat, qz, pz = self.ksvae_model(x)
            mu = qz.loc
            # z = self.ksvae_model.encoder(x)

            # vmf_mu, vmf_kappa, pose_theta = self.ksvae_model.shape_latent_to_vmf(z)
 
        output = self.classifier(mu)
        
        return output

    def reparameterize(self, mean, kappa):
        q_z = VonMisesFisher(mean, kappa)
        p_z = HypersphericalUniform(self.hypersphere_dim)

        return q_z, p_z         

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.ce(logits, y)
        # loss = F.nll_loss(logits, y)
        # norm_loss = torch.sum((torch.linalg.norm(self.shape_linear.weight, dim=1, keepdim=True) - 1.)**2) + torch.sum((torch.linalg.norm(self.appearance_linear.weight, dim=1, keepdim=True) - 1.)**2)
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss
        # return loss + norm_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.ce(logits, y)
        acc = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        # with torch.no_grad():
            # print(torch.linalg.norm(self.shape_linear.weight, dim=1, keepdim=True).squeeze())
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", avg_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.ce(logits, y)
        acc = self.accuracy(logits, y)
        return {"test_loss": loss, "test_accuracy": acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        self.log("test_loss", avg_loss)
        self.log("test_accuracy", avg_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}