"""
compute.py
================================
Classes for signature computation.
"""
from definitions import *
import numpy as np
import torch
import signatory
from src.features.signatures.functions import leadlag_slice


class DatasetSignatures():
    """ Signature computation method to work on TimeSeriesDatasets. """
    def __init__(self, augmentations, window, depth, logsig=False, leadlag=False, nanfill=True):
        self.augmentations = augmentations
        self.window = window
        self.depth = depth
        self.logsig = logsig
        self.leadlag = leadlag
        self.nanfill = nanfill

    def augment(self, data):
        for augmentation in self.augmentations:
            data = augmentation.transform(data)
        return data

    def transform(self, data):
        # Replace nans with zeros during this calculation
        if self.nanfill:
            nanmask = np.isnan(data).sum(axis=2) > 0
            data[nanmask] = 0

        # Apply augmentations
        aug_data = self.augment(data)

        # Compute the rolling signature with options
        signatures = RollingSignature(window=self.window, depth=self.depth, logsig=self.logsig).transform(aug_data)

        # Refill with nans
        if self.nanfill:
            signatures[nanmask] = np.nan

        # Keep only the useful leadlag slices
        if self.leadlag:
            signatures[:, leadlag_slice(), :]

        return signatures



class RollingSignature():
    """Computes signatures from rolling windows.

    Given input paths of shape [N, L, C] and a window size W. This computes the signatures over every possible interval
    of length W in the path length dimension (dim=1).

    Example:
        If the path is of shape [1, 5, 2] and W = 3 then we compute the signatures of each of:
            path[1, 0:3, 2], path[1, 1:4, 2], path[1, 2:5, 2]
        and then stack together and add some initial nan rows to illustrate a window of sufficient did not exist yet (if
        specified).
    """
    def __init__(self, window, depth, logsig=False, return_same_size=True):
        """
        Args:
            window (int): Length of the rolling window.
            depth (int): Signature depth.
            logsig (bool): Set True for a logsignature computation.
            return_same_size (bool): Set True to return a signature of the same size as the input path. This is achieved
                                     by addind a tensor of nans of size [N, window, signature_channels]
        # """
        self.window = window
        self.depth = depth
        self.logsig = logsig
        self.return_same_size = return_same_size

    @staticmethod
    def get_windows(path_len, window_len):
        """Gets the start and end indexes for the sliding windows.

        Args:
            path_len (int): Total length of the path.
            window_len (int): Desired window length.

        Returns:
            (list, list): List of start points and a list of end points.
        """
        end_points = np.arange(window_len, path_len)
        start_points = end_points - window_len
        return start_points, end_points

    @timeit
    def transform(self, paths):
        # Path info
        N, L, C = paths.shape[0], paths.shape[1], paths.shape[2]

        # Full signatures
        path_class = signatory.Path(paths, self.depth)

        # Logsig logic
        sig_func = path_class.logsignature if self.logsig else path_class.signature

        # Get indexes of the windows, apply the path_class signature function to each index.
        start_idxs, end_idxs = self.get_windows(L, self.window)
        signatures = torch.stack([sig_func(start, end) for start, end in zip(start_idxs, end_idxs)], dim=1)

        # Add nan values for the early times when signatures could not be computed
        if self.return_same_size:
            nans = float('NaN') * torch.ones((N, self.window, signatures.shape[2]))
            signatures = torch.cat((nans, signatures), dim=1)

        return signatures



if __name__ == '__main__':
    from definitions import *
    from sklearn.pipeline import Pipeline
    from src.features.signatures.augmentations import *

    # Load the data
    dataset = load_pickle(DATA_DIR + '/interim/preprocessed/dataset.dill', use_dill=True)
    data = dataset.data[[0], :, 0:2]

    augmentations = [
        AddTime(),
        LeadLag(),
    ]

    DatasetSignatures(augmentations, window=8, depth=3, logsig=True, nanfill=True).transform(data)

    x = LeadLag().transform(data)
    y = RollingSignature(window=8, depth=3, logsig=True).transform(x)

    y[:, slice(0, y.shape[1], 2), :]

