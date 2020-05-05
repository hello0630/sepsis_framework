"""
compute.py
================================
Classes for signature computation.
# TODO Make a separate rolling leadlag class
"""
from definitions import *
from copy import deepcopy
import numpy as np
import torch
import signatory
from src.features.signatures.augmentations import LeadLag
from src.features.signatures.functions import leadlag_slice


class DatasetSignatures():
    """Signature computation method to work with the TimeSeriesDataset class.

    Computes signatures over a rolling window on a time series dataset.

    Args:
        augmentations (list): A list of augmentations from src.features.signatures.augmentations. These will be applied
            to the path prior to signature computation.
        window (int): The length of the rolling window.
        depth (int): The depth to compute the signature to.
        nanfill (bool): Fill nans if they exist with zeros, this prevents a nan early on from breaking the signature
            computation later.
    """
    def __init__(self, augmentations, window, depth, logsig=False, nanfill=True, window_censor=True):
        self.augmentations = augmentations
        self.window = window
        self.original_window = window   # In case leadlag
        self.depth = depth
        self.logsig = logsig
        self.nanfill = nanfill
        self.window_censor = window_censor

        self.leadlag_slice = False
        if any([isinstance(x, LeadLag) for x in augmentations] + ['leadlag' in augmentations]):
            self.leadlag_slice = True
            self.window *= 2

    def augment(self, data):
        for augmentation in self.augmentations:
            data = augmentation.transform(data)
        return data

    def single_transform(self, data):
        # Fill to use signatory
        data_nanfill = deepcopy(data)
        data_nanfill[torch.isnan(data_nanfill)] = 0

        # Apply augmentations
        aug_data = self.augment(data_nanfill)

        # Compute the rolling signature with options
        signatures = RollingSignature(window=self.window, depth=self.depth, logsig=self.logsig).transform(aug_data)

        # If leadlag, only keep the relevant pieces
        if self.leadlag_slice:
            signatures = leadlag_slice(signatures)

        # Window and nan censoring
        # TODO This is a hack and is fairly slow. I do not know how to improve this though.
        for i in range(data.size(0)):
            censor = 0
            if self.window_censor:
                censor += self.original_window
            try:
                first_idx = np.argwhere(torch.isnan(data[i]).view(-1) == 0).view(-1)[0]
                censor += first_idx
                signatures[i, :censor, :] = np.nan
            except:
                signatures[i, :, :] = np.nan

        return signatures

    def transform(self, dataset, columns):
        signatures = []

        # Iterate
        for cols in columns:
            signatures.append(self.single_transform(dataset[cols]))

        signatures = torch.cat(signatures, dim=2)

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
        end_points = np.arange(1, path_len) + 1     # +1 for zero indexing
        start_points = np.arange(1, path_len) - window_len
        start_points[start_points < 0] = 0
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

        # Add a nan row since we cannot compute the signature of the first point
        signatures = torch.cat((float('nan') * torch.ones(N, 1, signatures.size(2)), signatures), dim=1)

        return signatures




if __name__ == '__main__':
    from definitions import *
    from sklearn.pipeline import Pipeline
    from src.features.signatures.augmentations import *

    a = torch.Tensor([1, 2, 3, 4, 5]).reshape(-1, 1)
    b = torch.ones_like(a)
    c = torch.cat((b, a), dim=1)