from definitions import *
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
    def __init__(self, data, labels, columns, ids=None):
        """
        Args:
            data (list): A list of variable length tensors. The shape must be [1, L_i, C] where L_i is the variable time
                         dimension, if list length is N this will create an [N, L_max, C] tensor.
            labels (torch.Tensor): Corresponding labels, to be input as a 1-D tensor.
            columns (list): List of column names of length C.
            ids (list): List of ids (integers preferred) of length N. Defaults to a range(0, N).
        """
        self.data = torch.nn.utils.rnn.pad_sequence(data, padding_value=np.nan, batch_first=True)
        self.labels = labels
        self.lengths = [d.size(0) for d in data]
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

    def drop(self, columns):
        """ Drops columns from the dataframe. """
        assert all([c in self.columns for c in columns])

        keep_idxs = [~x for x in self._col_indexer(columns)]
        self.columns = [c for c in self.columns if c not in columns]
        self.data = self.data[:, :, keep_idxs]

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
        params = {
            'data': self.data_to_list(),
            'labels': self.labels,
            'columns': self.columns,
            'ids': self.ids
        }
        save_pickle(params, loc)


if __name__ == '__main__':
    data, labels, columns, ids = generate_ts_data()
    dataset = TimeSeriesDataset(data, labels, columns, ids)
    dataset.save(DATA_DIR + '/interim/new_tsds/tsds_params.pickle')

