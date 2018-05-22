"""Sample script for training neural networks for classification.

This script provides multiple functions that demonstrate how to define,
train MLP architectures and how to optimize their hyperparameters.

Executing this script directly loads a sample classification dataset and
searches for the best hyperparameter values using random search.

Example:
    $ python basic_nn_classification.py
"""


import itertools
import numpy
import random

from experiment_dict import ExperimentDict

import keras
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

import sklearn
from sklearn.model_selection import train_test_split


################
# Util functions
################


def dict_product(d):
    """Cartesian product between dictionary values
    
    This function operates in the same way as itertools.product but takes
    as inputs a dictionary of iterables and will yield dictionaries of
    individual values.
    
    Args:
        d (dict): Dictionary overwhich to compute the cartesian product.

    Examples: 
        $ dict_product({'a' : [1, 2], 'b' : [3, 4]}) 
        
        Will yield an iterator containing the following elements :
            {'a' : 1, 'b' : 3}
            {'a' : 1, 'b' : 4} 
            {'a' : 2, 'b' : 3} 
            {'a' : 2, 'b' : 4}
    """
    for element in itertools.product(*d.values()):
        yield dict(zip(d.keys(), element))


def shuffle(iterable):
    """Returns the elements of an iterable in random order.
    
    Args:
        iterable (iterable): iterable to shuffle
        
    Returns:
        iterator: elements of the original iterable in random order

    NOTE : This will instantiate a list containing all the elements. If the
    iterable is a generator, all generated elements will be instantiated at
    once. This may end up taking a large amount of memory for very large
    number of elements in the generator.
    """
    items = list(iterable)
    random.shuffle(items)

    for element in items:
        yield element


def iterate_minibatches(X, y, batch_size=None):
    """Iterate over inputs and targets using specified batch size
    
    Args:
        X (array): Dataset inputs
        y (array): Dataset targets
        batch_size (int): Batch size. Default to size of the dataset.
    
    Returns:
        iterator: Each item in the iterator it a tuple (inputs, targets)
            respecting the specified batchsize. The last item may have
            less examples in it than the specified batch size if the size
            of the dataset is not a multiple of the specified batch size.
    """
    if batch_size is None:
        batch_size = len(X)

    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]


def onehot(y, min_val, max_val):
    """Convert vector of labels to one-hot matrix representation
    
    Args:
        y (1d array): Vector of class index labels (ex: 0, 1, 2, 3, ...)
        min_val (int): Lowest possible class index value
        max_val (int): Highest possible class index value
        
     
    """
    out = numpy.zeros((y.shape[0], max_val - min_val + 1))
    out[numpy.arange(y.shape[0]), y] = 1
    return out


################
# Main functions
################


def build_mlp(hyperparams, nb_input_feats, nb_classes):
    """Builds a MLP model using the Keras frameworké

    Args:
        hyperparams (dict): Hyperparameter describing the network's architecture
            Supported hyperparameters are : "layer_size", "batchnorm",
            "dropout", "nb_hidden_layers" and "l2_coeff"
        nb_input_feats (int): Number of input features to the first MLP layer
        nb_classes (int): Number of classes to predict

    Returns:
        Compiled Keras model
    """

    # Define model
    model = Sequential()

    model.add(Dense(hyperparams["layer_size"], input_dim=nb_input_feats, activation="relu"))

    if hyperparams["batchnorm"]:
        model.add(BatchNormalization())

    if hyperparams["dropout"]:
        model.add(Dropout(0.5))

    for hidden_layer in range(hyperparams["nb_hidden_layers"]):

        model.add(Dense(hyperparams["layer_size"], activation="relu",
                        kernel_regularizer=l2(hyperparams["l2_coeff"])))

        if hyperparams["batchnorm"]:
            model.add(BatchNormalization())

        if hyperparams["dropout"]:
            model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation="softmax"))

    # Compile model
    optimizer = Adam(lr=hyperparams["lr"], amsgrad=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model


def run_experiment(data, hyperparams, verbose=1):
    """Trains a MLP on provided dataset.

    Args:
        data (tuple): Dataset on which to run the experiment
            Structure of the tuple is
            (train_X, train_y), (valid_X, valid_y), (test_X, test_y)
            where xxxx_X are inputs (2d numpy array) and xxxx_y are labels
            (1d numpy array).
        hyperparams (dict): Hyperparameters for the experiment.
            Supported hyperparameters are : "layer_size", "batchnorm",
            "dropout", "nb_hidden_layers", "l2_coeff", "nb_epochs" and "seed".
        verbose (int): Amount of information to print during training.
            1 (default value) means all information. 0 means no information.
    Returns:
        dict:
        Keras model: State of the MLP at the latest epoch of the experiment
            Note : The latest epoch might not be the best epoch.
    """

    # Fix random seed for reproducibility
    random.seed(hyperparams["seed"])
    numpy.random.seed(hyperparams["seed"])

    # Unpack data
    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = data

    # Build model
    model = build_mlp(hyperparams, train_X.shape[1], train_y.max() + 1)

    # Convert labels to onehot format
    min_label, max_label = train_y.min(), train_y.max()
    train_y = onehot(train_y, min_label, max_label)
    valid_y = onehot(valid_y, min_label, max_label)
    test_y = onehot(test_y, min_label, max_label)

    # Main model training loop
    best_valid_accuracy = -numpy.inf
    best_model_results = None

    for epoch in xrange(hyperparams["nb_epochs"]):

        # Train model for 1 epochl
        model.fit(x=train_X, y=train_y, epochs=1, verbose=0)

        # Monitor performance on training, validation and test data
        train_loss, train_acc = model.evaluate(x=train_X, y=train_y, verbose=0)
        valid_loss, valid_acc = model.evaluate(x=valid_X, y=valid_y, verbose=0)
        test_loss, test_acc = model.evaluate(x=test_X, y=test_y, verbose=0)
        
        # Print current training status
        if verbose:
            print("Epoch %i" % epoch)
            print("  Train accuracy = %f, valid accuracy = %f" %
                  (train_acc, valid_acc))
            print("  Train loss = %f, valid loss = %f" %
                  (train_loss, valid_loss))

        # Log results if best model so far
        if valid_acc > best_valid_accuracy:

            best_valid_accuracy = valid_acc

            # Assemble dictionary containing all monitored metrics for train,
            # valid and test. Also add real number of epochs so we can retrain
            # the model in the future and get the same level of performance
            best_model_results = {"train_loss" : train_loss,
                                  "train_accuracy" : train_acc,
                                  "valid_loss" : valid_loss,
                                  "valid_accuracy" : valid_acc,
                                  "test_loss" : test_loss,
                                  "test_accuracy" : test_acc,
                                  "nb_epochs" : epoch + 1}

    # Return best results and latest model
    # WARNING : The best results may have been obtained before the last epoch.
    # If that is the case, the returned model will NOT be able to achieve
    # the results described in "best_model_metric"
    return best_model_results, model


def optimize_hyperparameters(data, possible_hyperparameter_values,
                             experiment_log_filename, nb_experiments=100):

    # Create object to log experiment results
    exp_dict = ExperimentDict(experiment_log_filename)

    # Create iterator for random hyperparameter search
    hyperparam_values_itt = shuffle(dict_product(possible_hyperparameter_values))

    # Run random hyperparameter search for the prescribed number of experiments
    for i in xrange(nb_experiments):

        # Pick the next hyperparameter values to try
        hyperparams = None
        for candidate_hyperparam_vals in hyperparam_values_itt:
            if not exp_dict.exp_in_log(candidate_hyperparam_vals):
                hyperparams = candidate_hyperparam_vals
                break

        # If not hyperparameter values could be selected, stop the process
        if hyperparams is None:
            break

        # Run experiment and print/log the results
        print("Training neural network with hyperparameters : %s" %
              str(hyperparams))
        exp_results, _ = run_experiment(data, hyperparams, verbose=0)

        print("Final results : %s \n" % str(exp_results))
        exp_dict.log_experiment(hyperparams, exp_results)

    # With the experiments done, go over the experiment log to find the
    # best model
    best_exp = exp_dict.get_best_experiment("valid_accuracy", higher_is_better=True)
    (best_exp_hyperparams, best_exp_result) = best_exp

    print("\n")
    print("Best experiment :")
    print("  Hyperparameters : %s" % best_exp_hyperparams)
    print("  Results : %s" % best_exp_result)


if __name__ == '__main__':

    # Load sample dataset
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    # Split test dataset into train, valid and test
    numpy.random.seed(123)
    X, test_X, y, test_y = train_test_split(X, y, test_size=0.20)
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.25)

    # Train neural networks on the data
    optimize_hyperparameters(
        data=((train_X, train_y), (valid_X, valid_y), (test_X, test_y)),
        possible_hyperparameter_values={
            "nb_epochs" : [500], "seed" : [123], "lr" : [1e-2, 1e-3, 1e-4, 1e-5],
            "dropout" : [True, False], "batchnorm" : [True, False],
            "nb_hidden_layers" : [1, 2, 3],
            "layer_size" : [16, 32, 64, 128, 256, 512],
            "l2_coeff" : [0, 1e-5, 1e-4, 1e-3, 1e-2]},
        experiment_log_filename="iris_experiment_log.pkl",
        nb_experiments=50)