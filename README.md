# Comparing Geo-VAEs
Code for training several different VAE models, namely:
* Original VAE
* Original autoencoder
* Hyperspherical VAE 
* SE2 equivariant VAE

All these above variants can be combined, so it is possible to train for example an equivariant hyperpsherical autoencoder model.
Datasets to train on currently include MNIST, STL10 and CIFAR10

## Examples of running code:

_Training a VAE model:_

``` python3 train.py --dataset=STL10 --batch_size=64 --hidden_dim 64 --k  --model_type=normal --variational ```

_Training an equivariant hyperspherical autoencoder model:_

``` python3 train.py --dataset=STL10 --batch_size=64 --hidden_dim 64 --k  --model_type=spherical --equivariant ```
