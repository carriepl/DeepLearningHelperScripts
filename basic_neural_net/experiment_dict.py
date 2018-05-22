"""Basic experiment-logging utility class."""

import numpy
import os
import pickle


class ExperimentDict(object):
    """ Basic experiment-logging utility class

    ExperimentDict represents a simple log in which experiment results can be
    logged, queried, saved to file and loaded back into memory.
    """

    def __init__(self, filename):
        """Constructor

        Build ExperimentDict object and load previous experiment data from
        specified filename.

        Args:
            filename (str): Name of file to use for loading/saving experiments
        """
        self.filename = filename
        if os.path.exists(filename) and os.path.isfile(filename):
            with open(filename, "rb") as f:
                self.experiments = pickle.load(f)
        else:
            self.experiments = []

    def log_experiment(self, hyperparams_dict, result_dict):
        """Log the hyperparameters and results of a single experiment

        Log the experiment in the ExperimentDict's internal log and save the
        latest version of the log to the filesystem.

        Args:
            hyperparams_dict (dict) : Hyperparameters of the experiment
                Keys are str representing the hyperparameter names and values
                represent the values of these hyperparameters.
            result_dict (dict) : Results of the experiment
                Keys are str representing the performance metrics and values
                represent the values of theses metrics.
        """
        self.experiments.append((hyperparams_dict, result_dict))
        self._save()

    def exp_in_log(self, hyperparams):
        """Returns boolean indicating whether an experiment with the same
        hyperparameters has already been logged.

        Args:
            hyperparams (dict) : Hyperparameters of the experiments
                Keys are str representing the hyperparameter names and values
                represent the values of these hyperparameters.

        Returns:
            bool: True if experiment has already been logged, False otherwise.
        """
        return hyperparams in [exp[0] for exp in self.experiments]

    def get_best_experiment(self, result_metric, higher_is_better=True):
        """Query best logged experiment according to specified metric.
        
        Args:
            result_metric (str): Performance metric to filter experiment by
            higher_is_better (bool): True if the best experiment is the one
                that maximizes the specified metric. False if it is the
                experiment that minimizes it. Defaults to True.
                
        Returns:
            tuple: Tuple containing hyperparams_dict and result_dict of the
                best experiment.
        """
        best_experiment = None
        best_metric_value = -numpy.inf if higher_is_better else numpy.inf

        for exp in self.experiments:
            try:
                exp_metric = exp[1][result_metric]
                if ((higher_is_better and exp_metric > best_metric_value) or
                    (not higher_is_better and exp_metric < best_metric_value)):

                    best_experiment = exp
                    best_metric_value = exp_metric
            except:
                pass

        if best_experiment is None:
            raise ValueError("No experiment has been logged with a result " +
                             "for this metric")

        return best_experiment

    def filter_experiments_by_metric(self, result_metric):
        """Query all experiments with specified metric logged.
        
        Returns the full data (hyperparameters and results) of all experiments
        for which a result has been logged for the specified metric.
        
        Args:
            result_metric (str): Metric to filter the experiments by.
        
        Returns:
            list of tuples: Experiment data (hyperparameters, results) of all
                matching experiment.
        """
        experiments_with_metric = []

        for exp in self.experiments:
            if result_metric in exp[1].keys():
                experiments_with_metric.append((exp[0], exp[1][result_metric]))

        return experiments_with_metric

    def _save(self):
        """Save experiment log to file."""
        with open(self.filename, "wb") as f:
            pickle.dump(self.experiments, f, pickle.HIGHEST_PROTOCOL)
