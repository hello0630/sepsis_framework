"""
preprocess.py
=================
Any preprocessing to apply to the data before we put it into a learning model.
"""
from definitions import *
import torch
from tqdm import tqdm
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


def make_timeseries_dataset():
    """ Turn the data into a time-series dataset. """
    # Get data
    df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')
    labels_utility = torch.Tensor(load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle'))
    labels_binary = torch.Tensor(load_pickle(DATA_DIR + '/processed/labels/binary.pickle'))

    # Remove unwanted cols
    df.drop(['time', 'SepsisLabel'], axis=1, inplace=True)
    columns = list(df.drop(['id'], axis=1).columns)

    # Convert df data to tensor form
    tensor_data = []
    ids = df['id'].unique()
    for id in tqdm(ids):
        data = df[df['id'] == id].drop('id', axis=1)
        tensor_data.append(torch.Tensor(data.values.astype(float)))

    # Create dataset
    dataset = TimeSeriesDataset(data=tensor_data, labels=labels_binary, columns=columns, ids=ids)

    # Useful to include binary_labels attr.
    save_pickle(dataset, DATA_DIR + '/interim/from_raw/dataset.dill', use_dill=True)

    return dataset



def main():
    # Make the dataset
    dataset = make_timeseries_dataset()
    scores = load_pickle(DATA_DIR + '/processed/labels/full_scores.pickle')

    # Dataset preprocessing
    dataset = preprocess_dataset(dataset)

    # Label preprocessing and pass info to dataset
    utility_value, labels = preprocess_labels(scores)
    dataset.labels_utility = utility_value

    save_pickle(dataset, DATA_DIR + '/interim/preprocessed/dataset.dill', use_dill=True)


if __name__ == '__main__':
    main()
