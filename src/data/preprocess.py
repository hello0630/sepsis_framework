"""
preprocess.py
=================
Any preprocessing to apply to the data before we put it into a learning model.
"""
from definitions import *
import torch
from src.data.dataset.dataset import TimeSeriesDataset
from src.data.dicts import feature_dict as feature_dict
from src.data.transformers import ForwardFill, DerivedFeatures
from src.features.transfomers import RollingStatistic, CountNonNan


def preprocess_dataset(dataset):
    """ Applied counts of num_variables, forward fill and basic derived feature transformations. """
    # Get counts before forward fill
    count_in_data = dataset[feature_dict['counts']]
    counts = CountNonNan().transform(count_in_data)
    column_names = [x + '_count' for x in feature_dict['counts']]
    dataset.add_features(counts, column_names)

    # Perform forward fill
    ffill = ForwardFill().transform(dataset.data)
    dataset.update_data(ffill)

    # Derive features
    dataset = DerivedFeatures().transform(dataset)

    return dataset


def preprocess_labels(scores):
    """ Preprocesses the labels to generate the labels needed for a binary classification problem. """
    # Compute utility value
    utility_value = torch.Tensor(scores[1] - scores[0])

    # Corresponding labels
    labels = torch.sign(utility_value)

    # Norm the utility_value
    utility_value = torch.abs(utility_value)

    return utility_value, labels


if __name__ == '__main__':
    # Load
    dataset = TimeSeriesDataset(**load_pickle(DATA_DIR + '/interim/from_raw/dataset.pickle'))
    scores = load_pickle(DATA_DIR + '/processed/labels/full_scores.pickle')

    # Dataset preprocessing
    dataset = preprocess_dataset(dataset)

    # Label preprocessing and pass info to dataset
    utility_value, labels = preprocess_labels(scores)
    dataset.binary_weights = utility_value
    dataset.binary_labels = labels

    save_pickle(dataset, DATA_DIR + '/interim/preprocessed/dataset.dill', use_dill=True)
