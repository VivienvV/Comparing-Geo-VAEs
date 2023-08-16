import torch
import argparse
import os
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
import io
import numpy as np
import PIL
import random
from typing import Sequence
from datetime import datetime



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--log', type=bool, default=True,
                        help='logging flag')
    parser.add_argument('--save_dir', type=str, default="saved models",
                        help='Directory in which to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--name', type=str, default="",
                        help='Name for the model')
    parser.add_argument('--train_augm', dest='train_augm', default=False, action='store_true',
                        help='use data augmentation (bool)')
    parser.add_argument('--test_augm', dest='test_augm', default=False, action='store_true',
                        help='use data augmentation (bool)')

    # Data parameters
    parser.add_argument('--dataset', type=str, default="STL10",
                        help='Data set')
    parser.add_argument('--root', type=str, default="VAES/data",
                        help='Data set location')
    parser.add_argument('--download', type=bool, default=True,
                        help='Download flag')

    # Model parameters
    parser.add_argument('--k', type=int, default=5,
                        help='Latent dimension')
    parser.add_argument('--N', type=int, default=8,
                        help='Order of cyclic group')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Base hidden dimension')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Kernel size')
    parser.add_argument('--depth', type=int, default=4,
                        help='Num layers')
    parser.add_argument('--kappa_min', type=float, default=1000.0,
                        help='kappa at the end of training')
    parser.add_argument('--kappa_max', type=float, default=1000.0,
                        help='kappa at the end of training')
    parser.add_argument('--model_type', type=str, default="spherical", choices=['spherical', 'normal'])
    parser.add_argument('--variational', dest='variational', default=False, action='store_true',
                        help='Use Variational Autoencoder')
    parser.add_argument('--equivariant', dest='equivariant', default=False, action='store_true',
                        help='Use equivariant autoencoder')
    parser.add_argument('--set_spreadloss', dest='set_spreadloss', default=False, action='store_true',
                        help='Use spread loss')

    # Parallel computing
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus to use (assumes all are on one node)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers in dataloader')

    args = parser.parse_args()

    now = datetime.now().strftime("%d-%m-%Y_%H-%M")

    if (args.kappa_min == args.kappa_max) and args.variational:
        raise Exception("Please lower kappa_min parameter to run as variational model") 
    if not args.equivariant:
        args.N = 1
    if args.model_type == 'normal':
        args.set_spreadloss = False
        

    print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print("|                                                                  Comparing Geometric VAEs                                                                |")
    print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(f'|    MODEL = {"Equivariant" if args.equivariant else "Non-equivariant"} {args.model_type} {"variational " if args.variational else ""}autoencoder')
    print('|    PARAMS: Spread Loss is {}, Train Augment is {}, Test Augment is {}'.format(args.set_spreadloss, args.train_augm, args.test_augm))    
    print('|    PARAMS: Hidden_dim: {}, Latent_dim: {}, Max_epochs: {}, Batch_size: {}, Learning Rate: {}, Kappa_min: {}, Kappa_max: {}'.format(args.hidden_dim, args.k, args.epochs, args.batch_size, args.lr, args.kappa_min, args.kappa_max))    
    print('|    DATASET: {}'.format(args.dataset))
    print('|    Started run on {}.'.format(now))
    print('--------------------------------------------------------------------------------------------------------------------------------------------------------------')

    model_name = ("eq_" if args.equivariant else "") +  ("S-" if args.model_type == 'spherical' else "N-") + ("VAE" if args.variational else "AE")

    if args.set_spreadloss:
        run_name = str(now) + '_' + model_name + str(args.N * args.k) + '_sumspreadloss'
    else:
        run_name = str(now) + '_' + model_name + str(args.N * args.k)
    args.name = run_name
    print(run_name)

    # Devices
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()
    if args.gpus > 1:
        args.batch_size = int(args.batch_size / args.gpus)

    input_dim = 1 if args.dataset ==  'MNIST' else 3
    
    # Initialize the model
    if args.model_type == 'normal' and args.equivariant:
        from VAES.models.SE2vae_Barrett import SE2VAE as Model
        model = Model(input_dim, args.k, args.N, args.hidden_dim, args.kernel_size, args.depth, args.variational, args.lr)
    elif args.model_type == 'normal' and not args.equivariant:
        from VAES.models.vae_Barrett import VanillaVAE as Model
        model = Model(input_dim, args.k, args.hidden_dim, args.kernel_size, args.depth, args.variational, args.lr)
    elif args.model_type == 'spherical' and args.equivariant:
        from VAES.models.ksvae_Barrett import VAES as Model
        model = Model(args.kappa_min, args.kappa_max, input_dim, args.k, args.N, args.hidden_dim, args.kernel_size, args.depth, args.variational, args.set_spreadloss, args.lr)
    elif args.model_type == 'spherical' and not args.equivariant:
        from VAES.models.svae_Barrett import SVAE as Model
        model = Model(args.kappa_min, args.kappa_max, input_dim, args.k, args.hidden_dim, args.depth, args.variational, args.set_spreadloss, args.lr)

    
    # Class for rotating image either 0, 90, 180 or 270 degrees
    class MyRotateTransform:
        def __init__(self, angles: Sequence[int]):
            self.angles = angles

        def __call__(self, x):
            angle = random.choice(self.angles)
            return TF.rotate(x, angle)

    normmean = (0.4914, 0.4822, 0.4465) if args.dataset=='CIFAR10' else (0.4914, 0.4822, 0.4465) if args.dataset=='STL10' else (0.5,) 
    normstd = (0.2023, 0.1994, 0.2010) if args.dataset=='CIFAR10' else (0.2471, 0.2435, 0.2616) if args.dataset=='STL10' else (0.5,)
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize(normmean, normstd),
                      # transforms.Pad((1,1,0,0)),
                      transforms.Pad((2)),
                      transforms.CenterCrop(model.spatial_size)
                     ]
    augm_list = [MyRotateTransform([0, 90, 180, 270])]

    if args.dataset == "MNIST":
        from torchvision.datasets import MNIST as DATASET
        input_dim = 1
        train_val_dataset = DATASET(args.root, train=True, download=args.download, transform = transforms.Compose(transform_list + augm_list if args.train_augm else transform_list))
        test_dataset = DATASET(args.root, train=False, download=args.download, transform = transforms.Compose(transform_list + augm_list if args.train_augm else transform_list))
        train_dataset, valid_dataset = torch.utils.data.random_split(train_val_dataset, [int(0.9 * len(train_val_dataset)), int(0.1 * len(train_val_dataset))])
    elif args.dataset == "STL10":
        from torchvision.datasets import STL10 as DATASET
        input_dim = 3
        train_dataset = DATASET(args.root, split='train+unlabeled', transform=transforms.Compose(transform_list + augm_list if args.train_augm else transform_list), download=args.download)
        test_dataset = DATASET(args.root, split='test', transform=transforms.Compose(transform_list + augm_list if args.train_augm else transform_list), download=args.download)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])
    elif args.dataset == "CIFAR10":
        from torchvision.datasets import CIFAR10 as DATASET
        input_dim = 3
        train_dataset = DATASET(args.root, train=True, transform=transforms.Compose(transform_list + augm_list if args.train_augm else transform_list), download=args.download)
        test_dataset = DATASET(args.root, train=False, transform=transforms.Compose(transform_list + augm_list if args.train_augm else transform_list), download=args.download)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])  
    else:
        raise Exception("Dataset could not be found")

    datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}
    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=args.num_workers)
        for split, dataset in datasets.items()}

    # logging
    if args.log:
        if args.set_spreadloss:
            logger = pl.loggers.WandbLogger(project=("(K)(S)(V)AE " + args.dataset), name=model_name + "_" + str(args.N * args.k) + '_sumspreadloss', config=args)
        else:
            logger = pl.loggers.WandbLogger(project=("(K)(S)(V)AE " + args.dataset), name=model_name + "_" + str(args.N * args.k), config=args)
        logger.experiment.config["effective_latent_dim"] = (args.N * args.k)
        
    else:
        logger = None

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    class Plotter(pl.Callback):

        def __init__(self, dataset, label):
            super().__init__()
            self.data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=args.num_workers)
            self.data_loader_iterator = iter(self.data_loader)
            self.label = label

        def on_validation_epoch_end(self, trainer, pl_module):
            input_imgs, _ = next(self.data_loader_iterator)
            input_imgs = input_imgs.to(pl_module.device)

            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            # Get the mean of the learned VMF distribution (included kendall points if k>1)
            if args.model_type == 'spherical':
                pts = reconst_imgs[1].loc
            else :
                pts = reconst_imgs[1]

            if args.equivariant:
                pts = pts.unflatten(-1, [-1, 2])

            # Unnormalize the images for plotting
            unmean = torch.tensor(normmean)
            unstd = torch.tensor(normstd)

            unnormalize = transforms.Normalize((-unmean / unstd).tolist(), (1.0 / unstd).tolist())
            unn_ims = unnormalize(input_imgs)
            unn_ims_recon = unnormalize(reconst_imgs[0]) 
            
            # The "raw" images
            ims = unn_ims.transpose(1, -1)
            ims_recon = unn_ims_recon.transpose(1, -1)

            # To to RGB and numpy
            ims = ims.expand(-1, -1, -1, 3).cpu().detach().numpy()
            ims_recon = ims_recon.expand(-1, -1, -1, 3).cpu().detach().numpy()

            buf = io.BytesIO()
            fig, axis = plt.subplots(nrows=pts.shape[0], ncols=2, figsize=(3, 8))
            plt.subplots_adjust(left=0.17,
                                bottom=0.05,
                                right=0.845,
                                top=0.95,
                                wspace=0.05,
                                hspace=0.0)
            for imnr in range(pts.shape[0]):
                axis[imnr][0].imshow(ims[imnr])
                axis[imnr][0].get_xaxis().set_visible(False)
                axis[imnr][0].get_yaxis().set_visible(False)
                axis[imnr][0].set_aspect("equal")
                axis[imnr][1].imshow(ims_recon[imnr])
                axis[imnr][1].get_xaxis().set_visible(False)
                axis[imnr][1].get_yaxis().set_visible(False)
                axis[imnr][1].set_aspect("equal")

            plt.savefig(buf)
            buf.seek(0)
            img = PIL.Image.open(buf)
            plt.close()
            trainer.logger.experiment.log({"random_" + self.label + "_recons": wandb.Image(img)})


    class CheckpointDuringTraining(pl.Callback):

        def __init__(self, name):
            super().__init__()
            self.name = name

        def on_validation_epoch_end(self, trainer, pl_module):
            trainer.save_checkpoint(f"./manual_checkpoints/{self.name}.ckpt")


    if args.name != "":
        run_name = args.name
    else:
        run_name = run_name


    # Do the training and testing
    callbacks = []
    if args.log:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
        callbacks.append(Plotter(datasets['valid'], "valid"))
        callbacks.append(Plotter(datasets['train'], "train"))
        callbacks.append(CheckpointDuringTraining(run_name))
        

    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=args.gpus, logger=logger, max_epochs=args.epochs, callbacks=callbacks,
                         gradient_clip_val=0.5)
    trainer.fit(model, dataloaders['train'], dataloaders['valid'])
    trainer.test(model, dataloaders['test'])
    trainer.save_checkpoint(f"./manual_checkpoints/{run_name}.ckpt")