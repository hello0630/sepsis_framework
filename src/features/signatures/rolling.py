"""
rolling.py
=============
Methods for computing signatures along rolling windows with different transformations.
"""
from definitions import *
import numpy as np
import torch
import signatory
from src.features.signatures.augmentations import LeadLag


class RollingMixin():
    """ Mixin for Rolling classes. """
    @staticmethod
    def get_windows(path_len, window_len):
        """Gets the start and end indexes for the sliding windows.

        Args:
            path_len (int): Total length of the path.
            window_len (int): Desired window length.

        Returns:
            (list, list): List of start points and a list of end points.
        """
        start_points = np.arange(0, path_len - window_len + 1)
        end_points = np.arange(window_len, path_len + 1)
        return start_points, end_points

    def add_window_nans(self, input_data, signature_data):
        # Account for window nans
        nans = np.nan * torch.ones(input_data.size(0), self.window - 1, signature_data.size(2))
        signatures = torch.cat((nans, signature_data), dim=1)
        return signatures

    def rolling_signature(self, data, window, logsig):
        # Get the length
        _, L, _ = data.size()

        # Create signatory path class
        path_class = signatory.Path(data, depth=self.depth)

        # Indexes to compute signatures on
        start_points, end_points = self.get_windows(L, window)

        # Ready the computation function
        sig_func = path_class.logsignature if logsig else path_class.signature

        # Compute the signatures
        signatures = torch.stack([sig_func(start, end) for start, end in zip(start_points, end_points)], dim=1)

        return signatures


class RollingPenOff(RollingMixin):
    """Signatures with the PenOff transformation applied along a rolling window.

    This method first augments an additional dimension to the path and computes the signatures along a rolling window
    utilising the signatory.Path class. We can then construct a tensor that contains the information to perform pendown
    and can append the pendown components on top of one another, utilise signatory.Path once more and keep only the
    pieces we need. A final signatory.signature_combine concatenates together.
    """
    def __init__(self, window=6, depth=3, logsig=False):
        self.window = window
        self.depth = depth
        self.logsig = logsig

    @timeit
    def transform(self, data):
        # Get sizes
        N, L, C = data.size()

        # Add the penoff dimension
        ones = torch.ones((N, L, 1))
        data_dim_aug = torch.cat((ones, data), dim=2)

        # Get the dimension augmented signatures
        signatures_dim_aug = self.rolling_signature(data_dim_aug, window=self.window, logsig=False)

        # Compute the pendown component
        _, end_points = self.get_windows(L, self.window)
        path_ends = data[:, end_points - 1, :]
        path_ends_dim_aug = torch.cat((torch.ones(N, len(end_points), 1), path_ends), dim=2)
        pendown = torch.repeat_interleave(path_ends_dim_aug, 3, dim=1)
        pendown[:, 1::3, 0] = 0
        pendown[:, 2::3, :] = 0

        # Get the pendown signatures
        pendown_rolling = self.rolling_signature(pendown, window=3, logsig=False)[:, 0::3, :]

        # Concat
        sig_dim = signatures_dim_aug.size(2)
        signatures_dim_aug = signatures_dim_aug.reshape(-1, sig_dim)
        pendown_rolling = pendown_rolling.reshape(-1, sig_dim)
        penoff = signatory.signature_combine(signatures_dim_aug, pendown_rolling, input_channels=C+1, depth=self.depth)

        if self.logsig:
            penoff = signatory.signature_to_logsignature(penoff, channels=C+1, depth=self.depth)
            sig_dim = penoff.size(1)

        # Remake
        penoff = penoff.reshape(N, -1, sig_dim)
        penoff_signatures = self.add_window_nans(data, penoff)

        return penoff_signatures


class RollingLeadLag(RollingMixin):
    """ Rolling signature computation with the leadlag transform. """
    def __init__(self, window=6, depth=3, logsig=False):
        self.window = window
        self.depth = depth
        self.logsig = logsig

    def transform(self, data):
        # Apply the leadlag transform
        ll_data = LeadLag().transform(data)

        # Compute the signatures
        signatures = self.rolling_signature(ll_data, self.window * 2, logsig=self.logsig)

        # Reduce the leadlag size

        # Add on the nans


        pass



if __name__ == '__main__':
    dataset = load_pickle(DATA_DIR + '/interim/preprocessed/dataset.dill')
    data = torch.rand(2, 10, 2)
    signatures = RollingLeadLag().transform(data)

    RollingPenOff().transform(dataset['SBP'])
