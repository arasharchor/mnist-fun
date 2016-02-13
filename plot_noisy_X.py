import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
from additional_functions import add_gaussian_noise
from keras_model import set_basic_model_param, load_and_format_mnist_data


def show_1_noisy_X_example(noise_stddevs, X_train):
    fig = plt.figure(figsize=(8, 1))
    outer_grid = gridspec.GridSpec(1, 13, wspace=0.0, hspace=0.0)
    pylab.xticks([])
    pylab.yticks([])
    for i, noise_stddev in zip(range(13), noise_stddevs[::8]):
        X_train_noisy = add_gaussian_noise(X_train, 0, noise_stddev)
        ax = plt.Subplot(fig, outer_grid[i])
        ax.imshow(X_train_noisy[0].reshape((28,28)), cmap=plt.cm.Greys)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    plt.show()


def show_all_noisy_X_example(noise_stddevs, X_train, y_train):
    fig = plt.figure(figsize=(10,10))
    outer_grid = gridspec.GridSpec(10, 13, wspace=0.0, hspace=0.0)
    pylab.xticks([])
    pylab.yticks([])
    first_ind_of_each_num = {i: np.where(y_train == i)[0][0] for i in range(10)}
    for col_ind, noise_stddev in zip(range(13), noise_stddevs[::8]):
        X_train_noisy = add_gaussian_noise(X_train, 0, noise_stddev)
        for row_ind in range(10):
            ind_to_plot = col_ind + row_ind * 13
            ax = plt.Subplot(fig, outer_grid[ind_to_plot])
            first_ind_of_this_num = first_ind_of_each_num[row_ind]
            ax.imshow(X_train_noisy[first_ind_of_this_num].reshape((28,28)), 
                      cmap=plt.cm.Greys)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

if __name__ == '__main__':
    model_param = set_basic_model_param(0)
    noise_stddevs = np.linspace(0, 192, 97)
    X_train, y_train, X_test, y_test = load_and_format_mnist_data(model_param)
    show_all_noisy_X_example(noise_stddevs, X_train, y_train)
