import numpy as np
from keras_model import *
from additional_functions import *
import matplotlib  # necessary to save plots remotely; comment out if local
matplotlib.use('Agg')  # comment out if local
import matplotlib.pyplot as plt
import pandas as pd


def train_models_on_noisy_X(noise_stddevs):
    model_param = set_basic_model_param(0)
    X_train, y_train, X_test, y_test = load_and_format_mnist_data(model_param,
                                            categorical_y=True)
    for noise_stddev in noise_stddevs:
        print 'Training models with noise lev of {}'.format(noise_stddev)
        model_param = set_basic_model_param(noise_stddev)
        noisy_X_train = add_gaussian_noise(X_train, mean=0, stddev=noise_stddev)
        model = compile_model(model_param)
        fit_and_save_model(model, model_param, noisy_X_train, y_train, 
                           X_test, y_test)


def calc_all_classwise_accs(noise_stddevs):
    model_param = set_basic_model_param(noise_stddevs[0])
    X_train, y_train, X_test, y_test = load_and_format_mnist_data(model_param, 
                                                categorical_y=False)
    unique_classes = np.unique(y_test)
    classwise_accs = {unique_class: [] for unique_class in unique_classes}
    for noise_stddev in noise_stddevs:
        print 'Calculating classwise accs for model with noise lev of {}'.format(noise_stddev)
        model = load_model('models/KerasBaseModel_v.0.1_{}'.format(noise_stddev))
        classwise_accs_to_add = predict_classwise_top_n_acc(model, X_test,
                                                            y_test)
        for elt in classwise_accs_to_add.keys():
            classwise_accs[elt].append(classwise_accs_to_add[elt])
    return classwise_accs


def plot_acc_vs_noisy_X(noise_stddevs, classwise_accs):
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
    plt.savefig('classwise_accuracy_vs_noisy_X_jet_3.png', dpi=200)

if __name__ == '__main__':
    noise_stddevs = np.linspace(0, 192, 97)
    #train_models_on_noisy_X(noise_stddevs)
    classwise_accs = calc_all_classwise_accs(noise_stddevs)
    plot_acc_vs_noisy_X(noise_stddevs, classwise_accs)
