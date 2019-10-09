import gzip
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from vae_keras import VAE
import sys
from config import Config as config

def load_mnist():
    path = 'E:\\work-space\\data_download\\mnist.pkl.gz'
    with gzip.open(path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set[0][:100], train_set[1][:100]

def show_img(arr, name):
    fig = plt.figure(figsize=(8,8))
    for i in range(1, 101):
        fig.add_subplot(10, 10, i)
        plt.imshow(arr[i-1].reshape((28,28)))
    plt.savefig(name)
    # plt.show()

def main():
    x, y = load_mnist()
    # show_img(x, 'ori')
    vae = VAE(784, config.latent_dim)
    vae.fit(x)
    x_encoded = vae.reconstruct(x)
    show_img(x_encoded, 'rec')
    epsilon_std = np.random.normal(0, 1, (100, config.latent_dim))
    z = []
    for i in range(-10, 10, 2):
        for j in range(-10, 10, 2):
            z.append(list(range(i+j, i+j+config.latent_dim)))
    z = np.array(z).reshape((100,config.latent_dim)) * epsilon_std
    x_gen = vae.generate(z)
    show_img(x_gen, "gen")

if __name__ == '__main__':
    main()
