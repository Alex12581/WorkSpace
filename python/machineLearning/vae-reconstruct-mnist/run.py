import gzip
import pickle
import math
import matplotlib.pyplot as plt

from vae_keras import VAE
import sys
arg = sys.argv[-1]

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
    show_img(x, 'ori')
    vae = VAE(784, 50)
    vae.fit(x, epochs=200)
    with open("model.pkl", "wb") as f:
        pickle.dump(vae, f, pickle.HIGHEST_PROTOCOL)
    x_encoded = vae.reconstruct(x)
    show_img(x_encoded, 'rec.png')

if __name__ == '__main__':
    main()
