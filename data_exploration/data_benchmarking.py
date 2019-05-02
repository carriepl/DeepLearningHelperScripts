"""Data benchmark with sklearn.

This script provides methods for benchmarking standardized datasets using
common machine learning models implemented by the Scikit-Learn library.

The functions classification_benchmark() and regression_benchmark() take as
input a dataset already separated into a training, a validation and a test set
and, respectively, try out multiple classification and regression models on the
data. For each model considered, basic hyper-parameter optimisation is performed
through grid-search.

Executing this script directly will demonstrate the script's usage by obtaining,
processing and benchmarking the IRIS flower identification dataset
(classification task).

Example:

    $ python benchmark.py

"""


import itertools
import numpy
import random

import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR


def dict_product(d):
    """ Computes cartesian product over dictionary with iterables values.

    This function operates in the same way as itertools.product but takes
    as inputs a dictionary of iterables and will yield dictionaries of
    individual values.

    Ex: dict_product({'a' : [1, 2], 'b' : [3, 4]}) will yield :
        {'a' : 1, 'b' : 3}
        {'a' : 1, 'b' : 4} 
        {'a' : 2, 'b' : 3} 
        {'a' : 2, 'b' : 4}
    """
    for element in itertools.product(*d.values()):
        yield dict(zip(d.keys(), element))


def accuracy(preds, targets):
    """ Computes accuracy between provided predictions and targets."""
    return (preds == targets).mean()


def MSE(preds, targets):
    """ Computes mean squared error between provided predictions and targets."""
    return ((preds - targets) ** 2).mean()


def benchmark_rbf_svm(model_class, data, model_selection_metric,
                      minimize_metric, seed=123):
    """ Benchmark a sklearn RBF SVM model on a provided dataset.
    
    This function operates like the benchmark_model() function but is more
    efficient for RBF SVM models. This is because it reuses the training data's
    similarity matrix between sets of compatible hyper-parameter values instead
    of constantly recomputing them. RBF SVM models could be benchmarked using
    the benchmark_model() function but it would be much less efficient.

    Args:
        model_class (class): Class of sklearn RBF SVM model to benchmark.
        data (tuple): Data to train the model on.
            Data is expected to fit the following format :
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
            train_x, valid_x and test_x are, respectively, the inputs of the
            training, validation and test set. Each takes the form of a numpy
            2d array of float32 values. train_y, valid_y and test_y are the
            targets and take the form of 1d arrays of float32 values.
        model_selection_metric (fct): Executable which takes two arguments, an
            array of predictions and an array of targets, and computes
            a performance metric from them.
        minimize_metric (bool): True if the specified model_selection_metric
            should be minimized, False if it should be maximized.
        seed (int): Integer value with which to seed random number generators
            to ensure reproducibility. Defaults to 123.
    """

    # Unpack data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data

    # If there are too many training examples, only use a random subset to 
    # compute the similarity matrix
    if len(train_x) > 1000:
        idx_ref_examples = range(len(train_x))
        numpy.random.shuffle(idx_ref_examples)
        rbf_kernel_ref = train_x[idx_ref_examples[:1000]]
    else:
        rbf_kernel_ref = train_x

    # Perform hyper-parameter optimisation
    print("Exploring hyperparameter values")
    best_hyperparams = None
    best_performance = numpy.inf if minimize_metric else -numpy.inf
    best_model = None

    for gamma in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100]:

        # Precomputed the transformed data to save time
        train_x_rbf = sklearn.metrics.pairwise.rbf_kernel(X=train_x,
                                                          Y=rbf_kernel_ref,
                                                          gamma=gamma)
        valid_x_rbf = sklearn.metrics.pairwise.rbf_kernel(X=valid_x,
                                                          Y=rbf_kernel_ref,
                                                          gamma=gamma)
        test_x_rbf = sklearn.metrics.pairwise.rbf_kernel(X=test_x,
                                                         Y=rbf_kernel_ref,
                                                         gamma=gamma)

        for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            hyperparams = {"gamma" : gamma, "C" : c}

            numpy.random.seed(seed)

            # Instantiate the model
            model = model_class(C=c)

            # Train the model
            model.fit(train_x_rbf, train_y)

            # Evaluate model
            performance = model_selection_metric(model.predict(valid_x_rbf), valid_y)

            if ((minimize_metric and performance < best_performance) or
                    (not minimize_metric and performance > best_performance)):
                best_hyperparams = hyperparams
                best_performance = performance
                best_model = model

            print("  For hyperparameters %s, validation performance : %f" %
                (str(hyperparams), performance))

    # Evaluate the best model on the test data
    test_performance = model_selection_metric(model.predict(test_x_rbf), test_y)
    print("Evaluating best model")
    print("  Best hyperparameters : %s" % str(best_hyperparams)) 
    print("  Test performance of best model : %f" % test_performance)
    print("\n")


def benchmark_model(model_class, iter_hyperparams, data,
                    model_selection_metric, minimize_metric, seed=123):
    """ Benchmark a sklearn model on a provided dataset.

    Args:
        model_class (class): Class of sklearn model to benchmark.
            A non-sklearn model class may be provided but it must implement
            the fit() and the predict() methods implemented by nearly all
            sklearn models.
        iter_hyperparams (iterable): Iterable of dictionaries. Each dictionary
            represents a set of hyper-parameter values to investigate.
            The dictionary keys represent the hyper-parameter names and the
            associated values represent the hyper-parameter values themselves.
        data (tuple): Data to train the model on.
            Data is expected to fit the following format :
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
            train_x, valid_x and test_x are, respectively, the inputs of the
            training, validation and test set. Each takes the form of a numpy
            2d array of float32 values. train_y, valid_y and test_y are the
            targets and take the form of 1d arrays of float32 values.
        model_selection_metric (fct): Executable which takes two arguments, an
            array of predictions and an array of targets, and computes
            a performance metric from them.
        minimize_metric (bool): True if the specified model_selection_metric
            should be minimized, False if it should be maximized.
        seed (int): Integer value with which to seed random number generators
            to ensure reproducibility. Defaults to 123.
    """

    # Unpack data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data

    # Perform hyper-parameter optimisation
    print("Exploring hyperparameter values")
    best_hyperparams = None
    best_performance = numpy.inf if minimize_metric else -numpy.inf
    best_model = None
    for hyperparams in iter_hyperparams:
    
        numpy.random.seed(seed)
    
        # Instantiate the model
        model = model_class(**hyperparams)
        
        # Train the model
        model.fit(train_x, train_y)

        # Evaluate model
        performance = model_selection_metric(model.predict(valid_x), valid_y)

        if ((minimize_metric and performance < best_performance) or
                (not minimize_metric and performance > best_performance)):
            best_hyperparams = hyperparams
            best_performance = performance
            best_model = model
        
        print("  For hyperparameters %s, validation performance : %f" %
              (str(hyperparams), performance))
    
    # Evaluate the best model on the test data
    test_performance = model_selection_metric(model.predict(test_x), test_y)
    print("Evaluating best model")
    print("  Best hyperparameters : %s" % str(best_hyperparams)) 
    print("  Test performance of best model : %f" % test_performance)
    print("\n")


def classification_benchmark(train_x, train_y, valid_x, valid_y, test_x, test_y):
    """ Benchmark multiple sklearn classification models on a provided dataset.

    Args:
        train_x (numpy 2d-array): Matrix of training inputs
        train_x (numpy 1d-array): Vector of integer training targets
        train_x (numpy 2d-array): Matrix of validation inputs
        train_x (numpy 1d-array): Matrix of integer validation targets
        train_x (numpy 2d-array): Vector of test inputs
        train_x (numpy 1d-array): Vector of integer test targets
    """
    
    print("Benchmarking constant model (always predicts most likely class)")
    benchmark_model(model_class=DummyClassifier,
                    iter_hyperparams=dict_product({'strategy' : ['most_frequent']}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=accuracy, minimize_metric=False)

    print("Benchmarking Gaussian Naive Bayes model")
    benchmark_model(model_class=GaussianNB,
                    iter_hyperparams=dict_product({'priors' : [None]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=accuracy, minimize_metric=False)

    print("Benchmarking K-nearest-neighbours model")
    benchmark_model(model_class=KNeighborsClassifier,
                    iter_hyperparams=dict_product({'n_neighbors' : [1, 2, 3, 5, 10, 25, 50, 100],
                                                'weights' : ["uniform", "distance"]}),
                    data=((train_x[:10000], train_y[:10000]), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=accuracy, minimize_metric=False)

    print("Benchmarking Random Forest model")
    benchmark_model(model_class=RandomForestClassifier,
                    iter_hyperparams=dict_product({'n_estimators' : [1, 5, 10, 25, 50, 100, 200, 500, 1000]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=accuracy, minimize_metric=False)

    print("Benchmarking Adaboost Regression model")
    benchmark_model(model_class=AdaBoostClassifier,
                    iter_hyperparams=dict_product({'n_estimators' : [1, 5, 10, 25, 50, 100, 200]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=accuracy, minimize_metric=False)

    print("Benchmarking Linear SVM model")
    benchmark_model(model_class=LinearSVC,
                    iter_hyperparams=dict_product({'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=accuracy, minimize_metric=False)

    print("Benchmarking RBF SVM model")
    benchmark_rbf_svm(model_class=LinearSVC,
                      data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                      model_selection_metric=accuracy, minimize_metric=False)


def regression_benchmark(train_x, train_y, valid_x, valid_y, test_x, test_y):
    """ Benchmark multiple sklearn regression models on a provided dataset.

    Args:
        train_x (numpy 2d-array): Matrix of training inputs
        train_x (numpy 1d-array): Vector of training targets
        train_x (numpy 2d-array): Matrix of validation inputs
        train_x (numpy 1d-array): Matrix of validation targets
        train_x (numpy 2d-array): Vector of test inputs
        train_x (numpy 1d-array): Vector of test targets
    """
    
    print("Benchmarking constant model (always predicts the mean of training labels)")
    benchmark_model(model_class=DummyRegressor,
                    iter_hyperparams=dict_product({'strategy' : ['mean']}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=MSE, minimize_metric=True)

    print("Benchmarking LASSO model")
    benchmark_model(model_class=Lasso,
                    iter_hyperparams=dict_product({'alpha' : [0.0001, 0.001, 0.01, 0.1, 0.3, 1, 3, 10]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=MSE, minimize_metric=True)

    print("Benchmarking ElasticNet model")
    benchmark_model(model_class=ElasticNet,
                    iter_hyperparams=dict_product({'alpha' : [0.0001, 0.001, 0.01, 0.1, 0.3, 1, 3, 10],
                                                   'l1_ratio' : [0.1, 0.25, 0.5, 0.75, 0.9]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=MSE, minimize_metric=True)

    print("Benchmarking Ridge Regression model")
    benchmark_model(model_class=Ridge,
                    iter_hyperparams=dict_product({'alpha' : [0.0001, 0.001, 0.01, 0.1, 0.3, 1, 3, 10]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=MSE, minimize_metric=True)

    print("Benchmarking Adaboost Regression model")
    benchmark_model(model_class=AdaBoostRegressor,
                    iter_hyperparams=dict_product({'n_estimators' : [1, 5, 10, 25, 50, 100]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=MSE, minimize_metric=True)

    print("Benchmarking Random Forest model")
    benchmark_model(model_class=RandomForestRegressor,
                    iter_hyperparams=dict_product({'n_estimators' : [1, 5, 10, 25, 50, 100, 200, 500, 1000]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=MSE, minimize_metric=True)
    
    print("Benchmarking Linear SVM model")
    benchmark_model(model_class=LinearSVR,
                    iter_hyperparams=dict_product({'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}),
                    data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                    model_selection_metric=MSE, minimize_metric=True)

    print("Benchmarking RBF SVM model")
    benchmark_rbf_svm(model_class=LinearSVR,
                      data=((train_x, train_y), (valid_x, valid_y), (test_x, test_y)),
                      model_selection_metric=MSE, minimize_metric=True)



if __name__ == '__main__':
    """ Example of classification benchmark on the IRIS dataset
    """

    # Obtain IRIS dataset
    from sklearn.datasets import fetch_mldata
    dataset = fetch_mldata('iris')

    # IRIS dataset is sorted by label. Shuffle inputs and targets jointly before
    # separation into train/valid/test.
    examples = zip(dataset.data, dataset.target)
    random.shuffle(examples)
    data, target = zip(*examples)

    # Transfer data to numpy arrays for more efficient manipulation
    data = numpy.array(data)
    target = numpy.array(target)

    # Split data into train, validation and test
    train_x = data[:100]
    train_y = target[:100]
    valid_x = data[100:125]
    valid_y = target[100:125]
    test_x = data[125:]
    test_y = target[125:]

    # Run benchmark (the IRIS labels are classes, not continuous variables) so
    # it is a classification task.
    classification_benchmark(train_x, train_y, valid_x, valid_y, test_x, test_y)
    #regression_benchmark(train_x, train_y, valid_x, valid_y, test_x, test_y)
