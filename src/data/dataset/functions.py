import numpy as np
import torch
import pandas as pd


def dataset_assertions(self):
    """ Assertions that can be run to ensure class variables have matching sizes. """
    # Info
    N, _, C = self.data.size()
    S = sum(self.lengths)

    to_assert = [
        N == len(self.ids),
        S == len(self.labels),
        C == len(self.columns)
    ]

    assert all(to_assert), 'Sizes mismatch!'


def generate_ts_data(N=10, L_max=50, C=5):
    """ Fake data generator for TimeSeriesDataset for use in testing.

    Args:
        N (int): Number of time-series.
        L_max (int): Max time-series length.
        C (int): Number of channels.
    """
    # Setup
    lengths = [np.random.randint(1, 50) for i in range(N)]

    # Make data
    data = [torch.randn(l, C) for l in lengths]
    labels = np.random.randint(0, 2, sum(lengths))
    columns = ['col_{}'.format(i) for i in range(C)]
    ids = np.arange(0, N)

    return data, labels, columns, ids


def index_getter(full_list, idx_items):
    """Boolean mask for the location of the idx_items inside the full list.

    Args:
        full_list (list): A full list of items.
        idx_items (list/str): List of items you want the indexes of.

    Returns:
        list: Boolean list with True at the specified column locations.
    """
    # Turn strings to list format
    if isinstance(idx_items, str):
        idx_items = [idx_items]

    # Check that idx_items exist in full_list
    diff_cols = [c for c in idx_items if c not in full_list]
    assert len(diff_cols) == 0, "The following cols do not exist in the dataset: {}".format(diff_cols)

    # Actual masking
    col_mask = [c in idx_items for c in full_list]

    return col_mask


def tensor_to_frame(data, lengths, columns, ids):
    """Convert the data into a pandas DataFrame. This method maps to only include the final times.

    Args:
        data (torch.Tensor): Data of shape [N, L, C]
        lengths (list): The lengths to extract each tensor to.
        columns (list): Names of the columns.
        ids (list): ID names

    Returns:
        pd.DataFrame: A pandas dataframe.
    """
    # Make frame
    dfs = []
    for i, data in enumerate(data):
        length = lengths[i]
        frame = pd.DataFrame(data=data[:length, :].numpy(), columns=columns)
        frame['id'] = ids[i]
        dfs.append(frame)
    df = pd.concat(dfs)

    # Reorder and reindex
    df = df[['id'] + [x for x in df.columns if x != 'id']]
    df = df.reset_index().drop('index', axis=1)

    return df
