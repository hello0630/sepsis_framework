import warnings
from copy import deepcopy
import numpy as np


class MakePaths():
    """ Creates paths for each id from each timepoint given a number of lookback times.

    Input a dataframe:
        1. Loops over ids
        2. Pads the dataframe with num_lookback times so we can get paths close to t=0
        3. Gets paths of variables from each timepoint of lookback length
        4. Appends paths to a list and returns

    TODO allow for a method that does not pad, and just uses less timepoints near 0 time
    """
    def __init__(self, lookback=6, method='max', last_only=False):
        """
        :param lookback: The length of the lookback window
        :param last_only: Returns the last path only. This is needed for submission algos.
        """
        self.lookback = lookback
        self.method = method
        self.last_only = last_only

    @staticmethod
    def pad_start(data, num):
        """ Pads the data with the initial values appended num times """
        if len(data.shape) == 1:
            data = data.reshape(data.shape[0], 1)
        padding = np.array(np.repeat(data[0], num)).reshape(num, -1, order='F')
        padded_data = np.concatenate([padding, data])
        return padded_data

    def transform(self, data_list):
        # Ready to store all paths
        all_paths = []

        # For each id, find all lookback paths and give to all paths
        for data in data_list:
            data = self.pad_start(data, 1)
            id_paths = [data[max(0, x - self.lookback):x] for x in range(2, data.shape[0] + 1)]
            all_paths.extend(id_paths)

        # Final path if submission
        if self.last_only:
            all_paths = [all_paths[-1]]
        all_paths = deepcopy(all_paths)
        return all_paths


if __name__ == '__main__':
    from definitions import *
    dataset = load_pickle(DATA_DIR + '/interim/preprocessed/dataset.dill')
    data = dataset.data_to_list()
    hr_data = [x[:, dataset._col_indexer('HR')] for x in data]
    paths = MakePaths(lookback=8).transform(hr_data)
