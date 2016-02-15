import numpy as np
np.random.seed(1234)  # for reproducibility
import time
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist


def set_basic_model_param(model_info):
    ''' 
    INPUT:  None
    OUTPUT: (1) Dictionary of important values for formatting data and 
                compiling model. 

    For lightweight tuning of the model (ie. no change in overall structure) 
    it's easiest to keep all model parameters in one place.
    '''
    model_param = {'n_rows': 28, 
                   'n_cols': 28,
                   'n_chan': 1,
                   'n_classes': 10,
                   'n_epoch': 4,
                   'batch_size': 32,
                   'pool_size': 2,
                   'conv_size': 3,
                   'n_conv_nodes': 32,
                   'n_dense_nodes': 128,
                   'primary_dropout': 0.25,
                   'secondary_dropout': 0.5,
                   'model_build': 'v.0.1_{}'.format(model_info)}
    return model_param


def load_and_format_mnist_data(model_param, categorical_y=False):
    ''' 
    INPUT:  (1) Dictionary: values important for formatting data appropriately
            (2) boolean: make the y values categorical? Keras requires the
                shape (#labels, #unique_labels), ie. (10000, 10) to train the
                model. However, randomizing the y labels is more easily done
                before the y labels are made categorical, when they are
                still of shape (#labels,) ie. (10000,)
    OUTPUT: (1) 4D numpy array: the X training data, of shape (#train_images,
                #chan, #rows, #columns); for MNIST this is (60000, 1, 28, 28)
            (2) 1D numpy array: the training labels, y, of shape (60000,)
            (3) 4D numpy array: the X test data, of shape (#test_images, 
                #chan, #rows, #columns); for MNIST this is (10000, 1, 28, 28)
            (4) 1D numpy array: the test labels, of shape (10000,)

    This function loads the data and labels, reshapes the data to be in the 
    4D tensor shape that Keras requires (#images, #color_channels, 
    #rows, #cols) for training, and returns it. The method load_data() 
    returns the MNIST data, shuffled and split between train and test set.
    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    num_train_images, num_test_images = X_train.shape[0], X_test.shape[0]
    X_train = X_train.reshape(num_train_images, 
                              model_param['n_chan'], 
                              model_param['n_rows'],
                              model_param['n_cols'])
    X_test = X_test.reshape(num_test_images, 
                            model_param['n_chan'],
                            model_param['n_rows'],
                            model_param['n_cols'])
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    if categorical_y:
        y_train = np_utils.to_categorical(y_train, model_param['n_classes'])
        y_test = np_utils.to_categorical(y_test, model_param['n_classes'])
    return X_train, y_train, X_test, y_test


def compile_model(model_param):
    ''' 
    INPUT:  (1) Dictionary of model parameters
    OUTPUT: (1) Compiled (but untrained) Keras model
    '''
    model = Sequential()
    model_param_to_add = [Convolution2D(model_param['n_conv_nodes'], 
                                        model_param['conv_size'],
                                        model_param['conv_size'],
                                        border_mode='valid',
                                        input_shape=(model_param['n_chan'],
                                                     model_param['n_rows'],
                                                     model_param['n_cols'])),
                          Activation('relu'),
                          Convolution2D(model_param['n_conv_nodes'], 
                                        model_param['conv_size'],
                                        model_param['conv_size']),
                          Activation('relu'),
                          MaxPooling2D(pool_size=(model_param['pool_size'],
                                                  model_param['pool_size'])),
                          Dropout(model_param['primary_dropout']),
                          Flatten(),
                          Dense(model_param['n_dense_nodes']),
                          Activation('relu'),
                          Dropout(model_param['secondary_dropout']),
                          Dense(model_param['n_classes']),
                          Activation('softmax')]

    for process in model_param_to_add:
        model.add(process)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model


def fit_and_save_model(model, model_param, X_train, y_train, X_test, y_test):
    ''' 
    INPUT:  (1) Compiled (but untrained) Keras model
            (2) Dictionary of model parameters
            (3) 4D numpy array: the X training data, of shape (#train_images,
                #chan, #rows, #columns); for MNIST this is (60000, 1, 28, 28)
            (4) 1D numpy array: the training labels, y, of shape (60000,)
            (5) 4D numpy array: the X test data, of shape (#test_images, 
                #chan, #rows, #columns); for MNIST this is (10000, 1, 28, 28)
            (6) 1D numpy array: the test labels, of shape (10000,)
    OUTPUT: None, but the model will be saved to /models
    '''
    start = time.clock()
    model.fit(X_train, y_train, batch_size=model_param['batch_size'],
              nb_epoch=model_param['n_epoch'],
              show_accuracy=True, verbose=1,
              validation_data=(X_test, y_test))
    stop = time.clock()
    total_run_time = (stop - start) / 60.
    score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
    print 'Test score: {}'.format(score[0])
    print 'Test accuracy: {}'.format(score[1])
    print 'Total run time: {}'.format(total_run_time)

    model_name = 'KerasBaseModel_{}'.format(model_param['model_build'])
    path_to_save_model = 'models/{}'.format(model_name)
    json_file_name = '{}.json'.format(path_to_save_model)
    weights_file_name = '{}.h5'.format(path_to_save_model)
    if os.path.isfile(json_file_name) or os.path.isfile(weights_file_name):
        json_file_name = '{}_copy.json'.format(path_to_save_model)
        weights_file_name = '{}_copy.h5'.format(path_to_save_model)
        print 'Please rename the model next time to avoid conflicts!'
    json_string = model.to_json()
    open(json_file_name, 'w').write(json_string)
    model.save_weights(weights_file_name)
