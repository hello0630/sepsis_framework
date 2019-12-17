from definitions import *
import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset


class SepsisDataset(Dataset):
    # TODO change how this is saved so we dont have to rerun this file each time. Saved params loaded into this.
    def __init__(self, data, labels=None, columns=None, ids=None):
        self.data = torch.nn.utils.rnn.pad_sequence(data, padding_value=np.nan, batch_first=True)
        self.labels = labels
        self.lengths = [d.size(0) for d in data]
        self.columns = columns if isinstance(columns, list) else list(columns)
        self.ids = ids

        # Additional indexers
        self.loc = SepsisLocIndexer(dataset=self)

        # Add long form of ids
        ids = [[x] * y for x, y in zip(self.ids, self.lengths)]
        self.ids_long = np.array(list(itertools.chain.from_iterable(ids)))

    def __getitem__(self, cols):
        return self.data[:, :, self.col_indexer(cols)]

    def __setitem__(self, key, item):
        self.data[:, :, self.col_indexer(key)] = item

    def __len__(self):
        return self.data.size(0)

    def drop(self, cols):
        # Get indexes to remove
        mask = np.array([self.col_indexer(col) for col in cols]).sum(axis=0)
        keep_idxs = np.argwhere((1 - mask)).reshape(-1)

        # Update
        self.columns = [self.columns[i] for i in keep_idxs]
        self.data = self.data[:, :, keep_idxs]

    def id_labels(self):
        """ Converts the labels tensor to a list of label tensors with each list entry corresponding to an id. """
        return [self.labels[np.argwhere(self.ids_long == id).reshape(-1)] for id in self.ids]

    def add_features(self, data, columns=None):
        """Method for adding newly computed features to each column.

        If new features are wanted to be added to

        Args:
            data (torch.Tensor): Tensor of shape [N, L, C_new] where C_new are the new feature channels to be added.
            columns (list): List containing column names for each of the new features. If unspecified becomes a range.

        Returns:
            None
        """
        new_features = data.shape[2]

        # Logic if columns unspecified
        if columns is None:
            int_cols = [x for x in dataset.columns if isinstance(x, int)]
            if len(int_cols) == 0: int_cols = [-1]
            max_int = max(int_cols)
            columns = list(range(max_int + 1, max_int + 1 + new_features))

        # Error handling
        assert data.shape[0:2] == self.data.shape[0:2], 'Dataset data and input data are different shapes.'
        assert len(columns) == new_features, 'Input data has a different length to input columns.'
        assert isinstance(columns, list), 'Columns must be inserted as list type.'
        assert len(set(columns)) == len(columns), 'Column names are not unique.'

        # Update
        self.data = torch.cat((self.data, data), dim=2)
        self.columns.extend(columns)

    def update_data(self, data):
        """ method for updating the data only up to where the length is specified. the rest is left as nan. """
        assert data.shape == self.data.shape, ('input data shape {} != current data shape {}.'
                                               .format(data.shape, self.data.shape))
        for i in range(self.data.shape[0]):
            self.data[i, :self.lengths[i], :] = data[i, :self.lengths[i], :]

    def to_frame(self):
        """ Convert the data into a pandas DataFrame. This method maps to only include the final times. """
        # Make frame
        dfs = []
        for i, data in enumerate(self.data):
            length = self.lengths[i]
            frame = pd.DataFrame(data=data[:length, :].numpy(), columns=self.columns)
            frame['id'] = self.ids[i]
            dfs.append(frame)
        df = pd.concat(dfs)

        # Reorder and reindex
        df = df[['id'] + [x for x in df.columns if x != 'id']]
        df = df.reset_index().drop('index', axis=1)

        return df

    def _make_X(self):
        data = []
        for id, length in zip(self.ids, self.lengths):
            data.append(self.data[id, :length, :])
        return torch.cat(data)

    @timeit
    def to_ml(self):
        X, y = self._make_X(), self.labels
        assert X.size(0) == y.size(0), "An error has occured making the machine learning data, X and y are not the same size."
        return X, y


class SepsisLocIndexer():
    """ An emulation of pandas loc behaviour to work with the sepsis dataset. """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, query):
        """
        Args:
            query [slice, list]: Works like the loc indexer, e.g. [1:5, ['col1', 'col2']].
        """
        idx, cols = query

        # Handle single column case
        if isinstance(cols, str):
            cols = [cols]

        # Ensure cols exist and get integer locations
        col_mask = self.dataset.col_indexer(cols)

        if not isinstance(idx, slice):
            assert isinstance(idx, int), 'Either index with a slice (a:b) or an integer.'
            idx = slice(idx, idx+1)

        return self.dataset.data[idx, :, col_mask]


if __name__ == '__main__':
    # Get data
    df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')
    labels_utility = torch.Tensor(load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle'))
    labels_binary = torch.Tensor(load_pickle(DATA_DIR + '/processed/labels/binary.pickle'))

    # Remove unwanted cols
    df.drop(['time', 'SepsisLabel'], axis=1, inplace=True)
    columns = df.drop(['id'], axis=1).columns

    # Convert df data to tensor form
    tensor_data = []
    ids = df['id'].unique()
    for id in tqdm(ids):
        data = df[df['id'] == id].drop('id', axis=1)
        tensor_data.append(torch.Tensor(data.values))

    # Create dataset
    dataset = SepsisDataset(tensor_data, labels=labels_binary, columns=columns, ids=ids)

    # Useful to include binary_labels attr.
    dataset.labels_utility = labels_utility

    save_pickle(dataset, DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)

