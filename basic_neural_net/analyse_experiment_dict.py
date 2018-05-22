"""Sample script for experiment analysis

This script analyses the content of a previously saved ExperimentDict object
and produces violin plots to illustrate the relationship between each
hyperparameter and a user-specified metric of interest

Example:
    $ python analyse_experiment_dict.py my_exp_log.pkl valid_accuracy

"""

import matplotlib.pyplot as plot
import sys

from experiment_dict import ExperimentDict


def collapse_lists(key_list, val_list):
    """Collapses a list of keys and a list of values into a dictionary
    
    Takes as input a list of keys and a list of values. Together, these lists
    repesent multiple (key, value) pairs. The functions groups these pairs
    into a dictionary of lists.
    
    Exemple : 
        $ collapse_lists(['a', 'b', 'a', 'a'], [1, 2, 3, 4])
        >>> {'a':[1,3,4], 'b':[2]}
    
    Args:
        key_list (list): list of keys for the dictionary
        val_list (list): list of values for the dictionary
    
    Return:
        dict: Dictionary containing the collapsed data from both args.
    """

    # Group values by common key
    val_dict = {}
    for k, v in zip(key_list, val_list):
        if k in val_dict:
            val_dict[k].append(v)
        else:
            val_dict[k] = [v]
    
    # Convert to lists and sort by key and by value
    out_key_list = sorted(val_dict.keys())
    out_val_list = [sorted(val_dict[k]) for k in out_key_list]
            
    return out_key_list, out_val_list


if __name__ == '__main__':
    # Recover command-line arguments
    dict_filename = sys.argv[1]
    metric_for_plotting = sys.argv[2]

    # Recover list of experiments for plotting
    exp_dict = ExperimentDict(dict_filename)
    experiments = exp_dict.filter_experiments_by_metric(metric_for_plotting)

    # Get a list of all hyperparameters used in any experiment
    all_hyperparams = []
    [all_hyperparams.extend(exp[0].keys()) for exp in experiments]
    all_hyperparams = list(set(all_hyperparams))

    # For each hyperparameter, plot how the different values it can take
    # influence the final performance of the model
    for hyperparam in all_hyperparams:
        
        # Obtain the two data lists to plot
        hyperparam_values = [exp[0][hyperparam] for exp in experiments]
        metric_values = [exp[1] for exp in experiments]
        
        (collapsed_hyperparam_values,
         collapsed_metric_values) = collapse_lists(hyperparam_values, metric_values)

        # Plot the data
        plot.clf()
        plot.violinplot(collapsed_metric_values)
        plot.title("Relationship between hyperparameter %s and %s" % (hyperparam, metric_for_plotting))
        plot.xticks(range(1, 1 + len(collapsed_hyperparam_values)), collapsed_hyperparam_values)
        plot.show()
        
    print("Best experiment according to metric of interest")
    print(exp_dict.get_best_experiment(metric_for_plotting))
        
        
        
        
        

