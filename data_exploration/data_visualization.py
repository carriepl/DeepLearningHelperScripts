"""Data exploration/visualisation with sklearn

This script provides examples of basic data visualisation techniques using the
Scikit-Learn library.

The function data_exploration() takes a dataset as input and performs a basic
analysis of the distribution of target values and explores different
techniques to project the data into a 2D space for easier visualisation.

Executing this script directly will demonstrate the script's usage by obtaining,
processing and benchmarking the Diabetes dataset (regression task).

Example:

    $ python data_exploration.py

"""

import itertools
import matplotlib.pyplot as plot
import numpy

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding


def explore_projection(model, X, Y=None, discrete_labels=False, out_filename=None):
    """ Project data according to specified projection and save the result

    Args:
        model (class): Sklearn class implementing a projection
            Non-sklearn classes may be used but they must offer the methods
            fit() and transform(), or the method fit_transform() with the same
            interface as in sklearn.
        X (numpy 2d array): Inputs of the dataset to analyse
        Y (numpy 1d array): Labels of the dataset to analyse (if applicable).
            Defaults to None (no labels provided).
        discrete_labels (bool): True if the labels contained in `Y` are
            discrete. False (default) if they are continuous.
        out_filename (str): Path where to save the plot of the projected data
            Defaults to None (which will display the plot instead of saving it).
    """

    # If possible, fit the projection on a random subset of the data to ensure
    # it doesn't take too much computation.
    try:
        numpy.random.seed(123)
        subset_indices = numpy.arange(len(X))
        numpy.random.shuffle(subset_indices)
        model.fit(X[subset_indices[:5000]])

        transformed_X = model.transform(X)[:, :2]

    except AttributeError:
        # The fit() or the transform() method is not implemented. this means
        # this is a model that must be trained on all the data to be
        # transformed (like T-SNE). In this case, use the fit_transform()
        # interface.
        transformed_X = model.fit_transform(X)[:, :2]

    # Plot the transformed data (colored by Ys, if possible)
    plot.figure(figsize=(10, 8))
    if Y is None:
        plot.scatter(transformed_X[:1000, 0], transformed_X[:1000, 1], s=10)
    else:
        if discrete_labels:
            plot.scatter(transformed_X[:, 0], transformed_X[:, 1], s=10,
                         c=Y, cmap=plot.cm.get_cmap('nipy_spectral', Y.max() + 1),
                         alpha=0.5)
            plot.colorbar()
        else:

            plot.scatter(transformed_X[:, 0], transformed_X[:, 1], s=10,
                         c=Y, cmap=plot.cm.get_cmap('nipy_spectral'),
                         alpha=0.5)
            plot.colorbar()
    plot.xlabel("1st component")
    plot.ylabel("2nd component")

    # Plot/save model
    if out_filename is None:
        plot.show()
        plot.clf()
    else:
        print("  Saving projection plot to %s" % out_filename)
        plot.savefig(out_filename)

    return model


def data_exploration(X, Y=None, discrete_labels=False):
    """ Performs various analytics and projections of the data

    Args:
        X (numpy 2d array): Inputs of the dataset to analyse
        Y (numpy 1d array): Labels of the dataset to analyse (if applicable)
            Defaults to None (no labels provided).
        discrete_labels (bool): True if the labels contained in `Y` are
            discrete. False (default) if they are continuous.
    """

    # Plot the distribution of Ys (targets)
    if Y is not None:

        print("Analysing the distribution of targets")
        if discrete_labels:
            classes = numpy.arange(Y.min(), Y.max() + 1)
            nb_per_class = (classes[:, None] == Y).sum(1)
            plot.bar(classes, nb_per_class, align='center')
        else:
            hist, bins = numpy.histogram(Y, bins=10)
            center = (bins[:-1] + bins[1:]) / 2
            plot.bar(center, hist, width=0.8 * (bins[1] - bins[0]), align='center')

        plot.xlabel("Target value")
        plot.ylabel("Number of examples")
        plot.savefig("label_distribution.png")
        print("  Histogram saved to label_distribution.png\n")


    # Perform PCA on the data
    print("Exploring PCA projection")
    pca = explore_projection(PCA(min(X.shape[1], 10)), X, Y, discrete_labels, "pca.png")

    print("  Leading eigenvectors and proportion of explained variance")
    print("  %s" % str(pca.components_))
    print("  Variance : %s" % str(pca.explained_variance_ratio_))


    # Perform Isomap on the data
    print("\nExploring Isomap projection")
    isomap = explore_projection(Isomap(5), X, Y, discrete_labels,
                                "isomap_5.png")
    isomap = explore_projection(Isomap(20), X, Y, discrete_labels,
                                "isomap_20.png")
    isomap = explore_projection(Isomap(50), X, Y, discrete_labels,
                                "isomap_50.png")


    # Plot transformed data points (colored by target if applicable)
    print("\nExploring LLE projection")
    lle = explore_projection(LocallyLinearEmbedding(5), X, Y, discrete_labels,
                             "lle_5.png")
    


if __name__ == '__main__':
    """ Toy examples on various datasets."""

    # Regression example on the Diabetes dataset
    from sklearn.datasets import load_diabetes
    X, y = load_diabetes(return_X_y=True)
    data_exploration(X, y, discrete_labels=False)

    """
    # Classification example on the IRIS dataset
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    data_exploration(X, y, discrete_labels=True)
    """