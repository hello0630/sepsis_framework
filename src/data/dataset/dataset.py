from definitions import *
from tqdm import tqdm
import itertools
import numpy as np
import torch
from src.data.dataset.functions import *
from src.data.dataset.indexers import LocIndexer


class TimeSeriesDataset():
    """Main class for dealing with time-series data.

    This class has been built out of a desire to perform matrix operations time-series data where the time-dimension has
    variable length. This assumes we have N time-series each with C channels and length L_i, where L_i can vary between
    series. This class will create a tensor of shape [N, L_max, C] where L_max is the longest time-series in the data
    and provide methods for keeping track of the original time-series components, whilst allowing the data to be
    manipulated through matrix operations.
    """
    def __init__(self, data=None, labels=None, columns=None, ids=None, lengths=None):
        """
        Args:
            data (list): A list of variable length tensors. The shape must be [1, L_i, C] where L_i is the variable time
                         dimension, if list length is N this will create an [N, L_max, C] tensor.
            labels (torch.Tensor): Corresponding labels, to be input as a 1-D tensor.
            columns (list): List of column names of length C.
            ids (list): List of ids (integers preferred) of length N. Defaults to a range(0, N).
            lengths (list): The lengths of each time-series. This will be auto-computed if data is inserted as a list.
        """
        if isinstance(data, list):
            self.data = torch.nn.utils.rnn.pad_sequence(data, padding_value=np.nan, batch_first=True)
            self.lengths = [d.size(0) for d in data]
        elif isinstance(data, torch.Tensor):
            self.data = data
            self.lengths = lengths
        self.labels = labels
        self.columns = columns
        self.ids = ids if ids is not None else np.arange(ids)

        self._assertions()

        # Additional indexers
        self.loc = LocIndexer(dataset=self)

    def _assertions(self):
        """ Basic size assertions. """
        dataset_assertions(self)

    def _col_indexer(self, cols):
        """ Returns a boolean list marking the index of the columns in the full column list. """
        return index_getter(self.columns, cols)

    def __getitem__(self, cols):
        """ Will return the column much like pandas.

        Args:
            cols (list/str): A list of columns or a column name string.

        Returns:
            torch.Tensor: Tensor corresponding to the chosen columns.
        """
        return self.data[:, :, self._col_indexer(cols)]

    def __setitem__(self, key, item):
        """
        Args:
            key (list/str): Columns to overwrite.
            item (torch.Tensor): Data to overwrite with.

        Returns:
            None
        """
        self.data[:, :, self._col_indexer(key)] = item

    def __len__(self):
        return self.data.size(0)

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
        """Updates data in the dataframe preserving the nan structure.

        Given a data tensor of the same size as self.data, updates with the new data up to where the length is
        specified, the rest being left as nan.
        """
        assert data.shape == self.data.shape, ('input data shape {} != current data shape {}.'
                                               .format(data.shape, self.data.shape))
        for i in range(self.data.shape[0]):
            self.data[i, :self.lengths[i], :] = data[i, :self.lengths[i], :]

    def drop(self, columns):
        """ Drops columns from the dataframe. """
        assert all([c in self.columns for c in columns])

        keep_idxs = [~x for x in self._col_indexer(columns)]
        self.columns = [c for c in self.columns if c not in columns]
        self.data = self.data[:, :, keep_idxs]

    def get_long_ids(self):
        """ Returns the long form of the ids. That is such there is a corresponding id for each timepoint. """
        ids = [[x] * y for x, y in zip(self.ids, self.lengths)]
        return np.array(list(itertools.chain.from_iterable(ids)))

    def data_to_list(self):
        """ Converts the tensor data back onto list format. """
        tensor_list = []
        for i, l in enumerate(self.lengths):
            tensor_list.append(self.data[i, 0:l, :])
        return tensor_list

    def save(self, loc):
        """ Save method. Saves only the necessary components.

        Args:
            loc (str): The save location. Typically use .tsds extension.
        """
        keys = ['data', 'labels', 'columns', 'ids', 'lengths']
        save_dict = {key: self.__dict__[key] for key in keys}
        save_pickle(save_dict, loc)

    def to_frame(self):
        """ Turn the data into a dataframe. """
        return tensor_to_frame(self.data, self.lengths, self.columns, self.ids)

    def to_ml(self, output_ids=True):
        """ Outputs the data in machine learning form.

        Args:
            output_ids (bool): Set true to output the corresponding ids. Useful for doing cross validation.
        """
        # Get ML form
        X = torch.cat(self.data_to_list())
        y = self.labels

        # ID logic
        if output_ids:
            ids = self.get_long_ids()
            return X, y, ids
        else:
            return X, y


if __name__ == '__main__':
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
        tensor_data.append(torch.Tensor(data.values))

    # Create dataset
    dataset = TimeSeriesDataset(data=tensor_data, labels=labels_binary, columns=columns, ids=ids)

    # Useful to include binary_labels attr.
    dataset.save(DATA_DIR + '/interim/from_raw/dataset.pickle')


