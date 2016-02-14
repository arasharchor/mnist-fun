import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
from additional_functions import add_gaussian_noise
from keras_model import set_basic_model_param, load_and_format_mnist_data


def show_1_noisy_X_example(noise_stddevs, X_train, ind_to_display=0):
    ''' 
    INPUT:  (1) 1D numpy array: standard deviations of the gaussian noise to add
                to an example image from the X training data. Note that the image
                data has not yet been scaled from 0 to 1, but still has values
                between 0 and 255
            (2) 4D numpy array: X training data
            (3) integer: the index from X_train to display with noise over it
    OUTPUT: None
    
    This function displays one image with increasing levels of gaussian noisee
    on top of it.
    '''
    fig = plt.figure(figsize=(8, 1))
    outer_grid = gridspec.GridSpec(1, 13, wspace=0.0, hspace=0.0)
    pylab.xticks([])
    pylab.yticks([])
    for i, noise_stddev in zip(range(13), noise_stddevs[::8]):
        X_train_noisy = add_gaussian_noise(X_train, 0, noise_stddev)
        ax = plt.Subplot(fig, outer_grid[i])
        ax.imshow(X_train_noisy[ind_to_display].reshape((28,28)), cmap=plt.cm.Greys)
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
