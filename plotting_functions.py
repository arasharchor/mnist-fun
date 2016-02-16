import numpy as np
from keras_model import *
from additional_functions import *
#import matplotlib  # necessary to save plots remotely; comment out if local
#matplotlib.use('Agg')  # comment out if local
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab
from keras.utils import np_utils
import pandas as pd


def train_models_on_noisy_data(characteristic_noise_vals, X_or_y):
    ''' 
    INPUT:  (1) 1D numpy array: if on X, should be the standard deviations of
                the Gaussian noise being added; if on y, should be the 
                percentages of labels to be randomly changed
            (2) string: 'X' or 'y' corresponding to which data to make noisy
    OUTPUT: None, directly at least. All models will be saved to /models    

    This function loads the basic data, then loops through the characteristic
    noise values and trains models on those noisy data. Classwise accuracies 
    can then be calculated from these models. 
    '''
    model_param = set_basic_model_param(0)
    X_train, y_train, X_test, y_test = load_and_format_mnist_data(model_param,
                                            categorical_y=False)
    for cnv in characteristic_noise_vals:
        name_to_append = '{}_{}'.format(X_or_y, cnv)
        model_param = set_basic_model_param(name_to_append)
        if X_or_y == 'X':
            print 'Training models with noise lev of {}'.format(cnv)
            noisy_X_train = add_gaussian_noise(X_train, mean=0, stddev=cnv)
            y_train = np_utils.to_categorical(y_train, model_param['n_classes'])
            y_test = np_utils.to_categorical(y_test, model_param['n_classes'])
            model = compile_model(model_param)
            fit_and_save_model(model, model_param, noisy_X_train, y_train, 
                               X_test, y_test)
        if X_or_y == 'y':
            print 'Training models with {}% random labels'.format(cnv)
            noisy_y_train = add_label_noise(y_train, cnv)
            noisy_y_train = np_utils.to_categorical(noisy_y_train, 
                                                    model_param['n_classes'])
            y_test = np_utils.to_categorical(y_test, model_param['n_classes'])
            model = compile_model(model_param)
            fit_and_save_model(model, model_param, X_train, noisy_y_train, 
                               X_test, y_test)


def calc_all_classwise_accs(noise_stddevs):
    '''
    INPUT:  (1) 1D numpy array: The standard deviations of the Gaussian noise 
                being added to the data
    OUTPUT: (1) dictionary of lists: The accuracies over all standard deviations 
                for each digit in MNIST

    This function calculates the classwise accuracies as a function of the
    standard deviation of the Gaussian noise added to the X training data. 
    It isn't set up to handle the noisy y data, as classwise accuracies 
    do not make much sense to calculate when looking at the effect that
    randomizing some percentage of the labels has on the model performance. 
    '''
    model_param = set_basic_model_param(noise_stddevs[0])
    X_train, y_train, X_test, y_test = load_and_format_mnist_data(model_param, 
                                                categorical_y=False)
    unique_classes = np.unique(y_test)
    classwise_accs = {unique_class: [] for unique_class in unique_classes}
    for noise_stddev in noise_stddevs:
        print '''Calculating classwise accs for model characteristic noise value
                of of {}'''.format(x)
        name_to_append = '{}_{}'.format('X', noise_stddev)
        model = load_model('models/KerasBaseModel_v.0.1_{}'.format(name_to_append))
        classwise_accs_to_add = predict_classwise_top_n_acc(model, X_test,
                                                            y_test)
        for elt in classwise_accs_to_add.keys():
            classwise_accs[elt].append(classwise_accs_to_add[elt])
    return classwise_accs


def calc_raw_acc(characteristic_noise_vals, X_or_y):
    ''' 
    INPUT:  (1) 1D numpy array: if on X, should be the standard deviations of
                the Gaussian noise being added; if on y, should be the 
                percentages of labels to be randomly changed
            (2) string: 'X' or 'y' corresponding to which data was made noisy
                before training the models for which we are calculating the acc

    This function calculates the raw accuracy (over all classes) a series of
    models with different characteristic noise values.
    '''
    model_param = set_basic_model_param(0)    
    X_train, y_train, X_test, y_test = load_and_format_mnist_data(model_param, 
                                                categorical_y=False)
    accs = []
    for cnv in characteristic_noise_vals:
        print '''Calculating raw accuracy for models with a characteristic 
                 noise value of {}'''.format(cnv)
        name_to_append = '{}_{}'.format(X_or_y, cnv)
        model = load_model('models/KerasBaseModel_v.0.1_{}'.format(name_to_append))
        y_pred = model.predict_classes(X_test)
        acc_to_add = np.sum(y_pred == y_test) / float(len(y_test))
        accs += [acc_to_add]
    return accs 
    

def plot_acc_vs_noisy_X(noise_stddevs, classwise_accs, saveas):
    ''' 
    INPUT:  (1) 1D numpy array: The standard deviations of the Gaussian noise 
                being added to the data
            (2) dictionary of lists: The accuracies over all standard deviations 
                for each digit in MNIST (the output of calc_all_classwise_accs)
            (3) string: the name to save the plot
    OUTPUT: None. However, the plot will be saved at the specified location.

    Classwise accuracies will be plotted vs. the standard deviation of Gaussian
    noise added to the X training data. A rolling mean is applied to make the 
    plot readable; nans created by the rolling mean are filled with original
    values for completeness.
    '''
    unique_classes = sorted(classwise_accs.keys())
    color_inds = np.linspace(0, 1, len(unique_classes))
    for color_ind, unique_class in zip(color_inds, unique_classes):
        rolling_mean = pd.rolling_mean(np.array(classwise_accs[unique_class]),
                                       window=3, center=False)
        null_ind_from_rolling = np.where(pd.isnull(rolling_mean))[0]
        orig_vals_to_fill_nulls = np.array(classwise_accs[unique_class])[null_ind_from_rolling]
        rolling_mean[null_ind_from_rolling] = orig_vals_to_fill_nulls
        plt.plot(noise_stddevs, rolling_mean, 
                 color=plt.cm.jet(color_ind), label=str(unique_class))
    plt.xlabel('Standard Deviation of Gaussian Noise Added to Training Data')
    plt.ylabel('Accuracy')
    plt.legend(loc=3)
    plt.savefig('{}.png'.format(saveas), dpi=200)


def plot_acc_vs_noisy_y(percent_random_labels, accs, saveas):
    ''' 
    INPUT:  (1) 1D numpy array: The fraction of training labels randomized
            (2) list: The accuracies for each model (the output of 
                calc_raw_acc with 'y')
            (3) string: the name to save the plot
    OUTPUT: None. However, the plot will be saved at the specified location.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(percent_random_labels, accs, label='Model Accuracy on Test Set')
    ax.set_xlabel('Percent of Training Labels Randomized')
    ax.set_ylabel('Accuracy')
    x_tick_vals = ax.get_xticks()
    ax.set_xticklabels(['{:2.1f}%'.format(x * 100) for x in x_tick_vals])
    ax.set_xlim(0, .33)
    ax.set_ylim(0, 1)
    ax.axhline(.1, ls=':', color='k', label='Naive Guessing')
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    fig.savefig('{}.png'.format(saveas), dpi=200)


def show_1_noisy_X_example(noise_stddevs, X_train, ind_to_display=0):
    ''' 
    INPUT:  (1) 1D numpy array: standard deviations of the Gaussian noise to add
                to an example image from the X training data. Note that the image
                data has not yet been scaled from 0 to 1, but still has values
                between 0 and 255
            (2) 4D numpy array: X training data
            (3) integer: the index from X_train to display with noise over it
    OUTPUT: None, but the plot will show to screen
    
    This function displays one image (specified by ind_to_display) 
    with increasing levels of Gaussian noise on top of it.
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
    ''' 
    INPUT:  (1) 1D numpy array: standard deviations of the Gaussian noise to add
                to example images from the X training data. Note that the image
                data has not yet been scaled from 0 to 1, but still has values
                between 0 and 255. Hardcoded to work with the len of this array
                at 97 (the total number of models trained) such that taking 
                every 8th element results in 13 examples (which is the hardcoded
                number of columns for this function)
            (2) 4D numpy array: X training data
            (3) 1D numpy array: y training data: the first instance of each digit
                will be taken from these labels so that an example of each
                digit can be shown
    OUTPUT: None, but the plot will show to screen
    
    This function displays an example of each digit from the X training data
    with increasing levels of Gaussian noise on top of it. The function is 
    hardcoded such that 13 examples of increasing noise will be shown. 
    '''
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
            if ax.is_last_row():
                ax.set_xlabel('{}'.format(noise_stddev))
    plt.show()


def load_data_and_show_noisy_X():
    ''' 
    INPUT:  None
    OUTPUT: None, but the plot from show_all_noisy_X_example will show to screen
    
    This function loads the data and utilizes show_all_noisy_X_example to
    give an example of what the training data look like with increasing levels
    of Gaussian noise. 
    '''
    model_param = set_basic_model_param(0)
    noise_stddevs = np.linspace(0, 192, 97)
    X_train, y_train, X_test, y_test = load_and_format_mnist_data(model_param)
    show_all_noisy_X_example(noise_stddevs, X_train, y_train)
