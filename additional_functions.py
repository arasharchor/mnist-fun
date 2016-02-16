import numpy as np
from keras.models import model_from_json


def load_model(path_to_model):
    ''' 
    INPUT:  (1) String: The path to the saved model architecture and weights, 
                not including .json or .h5 at the end
    OUTPUT: (1) Trained and compiled Keras model
    '''
    json_file_name = '{}.json'.format(path_to_model)
    weights_file_name = '{}.h5'.format(path_to_model)
    model = model_from_json(open(json_file_name).read())
    model.load_weights(weights_file_name)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model


def predict_classwise_top_n_acc(model, X_test, y_test, n=1):
    ''' 
    INPUT:  (1) Trained and compiled Keras model
            (2) 4D numpy array: all the test set images
            (3) 1D numpy array: the corresponding test set labels
            (4) integer: if the true class is in the top n of predictions,
                count it as correct. n=1 provides the usual top-1 accuracy.
    OUTPUT: (1) Dictionary: the classes as keys, with corresponding 
                accuracies as values

    This function is able to calculate the top-n accuracy on a classwise basis.
    '''
    unique_classes = np.unique(y_test)
    y_test = y_test.reshape((y_test.shape[0], 1))
    probas = model.predict_proba(X_test, batch_size=32)
    top_n_guesses = np.fliplr(np.argsort(probas, axis=1))[:, :n]
    classwise_acc_dict = {}
    for unique_class in unique_classes:
        unique_class_locs = np.where(y_test == unique_class)[0]
        top_n_guesses_for_this_class = top_n_guesses[unique_class_locs]
        in_top_n = [1 if y_test[row_idx] in row
                    else 0
                    for row_idx, row 
                    in zip(unique_class_locs, top_n_guesses_for_this_class)]
        classwise_acc_dict[unique_class] = (np.sum(in_top_n) / 
                                            float(len(unique_class_locs)))
    return classwise_acc_dict
    

def add_gaussian_noise(X_train, mean, stddev):
    ''' 
    INPUT:  (1) 4D numpy array: all raw training image data, of shape 
                (#imgs, #chan, #rows, #cols)
            (2) float: the mean of the Gaussian to sample noise from
            (3) float: the standard deviation of the Gaussian to sample
                noise from. Note that the range of pixel values is
                0-255; choose the standard deviation appropriately. 
    OUTPUT: (1) 4D numpy array: noisy training data, of shape
                (#imgs, #chan, #rows, #cols)
    '''
    n_imgs = X_train.shape[0]
    n_chan = X_train.shape[1]
    n_rows = X_train.shape[2]
    n_cols = X_train.shape[3]
    if stddev == 0:
        noise = np.zeros((n_imgs, n_chan, n_rows, n_cols))
    else:
        noise = np.random.normal(mean, stddev/255., 
                                 (n_imgs, n_chan, n_rows, n_cols))
    noisy_X = X_train + noise
    clipped_noisy_X = np.clip(noisy_X, 0., 1.)
    return clipped_noisy_X


def add_label_noise(y_train, percent_to_randomize):
    ''' 
    INPUT:  (1) 1D numpy array: y training data
            (2) float: the fraction of data to randomize the labels for
    OUTPUT: (1) 1D numpy array: noisy y training data
    '''
    n_labels = y_train.shape[0]
    n_to_randomize = n_labels * percent_to_randomize
    n_random_labels = np.random.randint(low=0, high=10, size=n_to_randomize)
    ordered_ind = np.arange(n_labels)
    permuted_ind = np.random.permutation(ordered_ind)
    random_ind = permuted_ind[:n_to_randomize]
    y_train[random_ind] = n_random_labels
    return y_train
