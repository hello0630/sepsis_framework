from definitions import *
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.functions import pytorch_rolling
import warnings


class FeaturePipeline():
    """Basic feature transformation pipeline.

    This class currently works with feature transformers that contain a transform method and do not require fitting.
    Takes a steps argument that should consist of feature transformers that take a torch.Tensor of shape [N, L, C] and
    output data of shape [N, L, new_features].
    """
    def __init__(self, steps):
        """
        Args:
            steps (list): List of tuples where each tuple element of the list is of the form (name, cols, transformer).
                          'name' is the name of the transformation step, 'cols' are the column names the transformer is
                          to act on, and 'transformer' is a class containing a transform method that acts on a 3D
                          torch.Tensor to return transformed features.
        """
        self.steps = steps

    def transform(self, dataset):
        outputs = []
        for name, cols, transformer in self.steps:
            output = transformer.transform(dataset[cols])
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-1)
        return outputs


class RollingStatistic():
    """Applies statistics to a rolling window along the time dimension of a tensor.

    Given an input tensor of shape [N, L, C] and a specified window size, W, this function first expands the tensor to
    one of shape [N, L, C, W] where W has expanded out the time dimension (this here is the L-dimension). The final
    dimension contains the most recent W time-steps (with nans if not filled). The specified statistic is then computed
    along this W dimension to give the statistic over the rolling window.
    """
    def __init__(self, statistic, window_length, step_size=1, func_kwargs={}):
        """
        # TODO implement a method that removes statistics that contained insufficient data.
        Args:
            statistic (str): The statistic to compute.
            window_length (int): Length of the window.
            step_size (int): Window step size.
        """
        self.statistic = statistic
        self.window_length = window_length
        self.step_size = step_size
        self.func_kwargs = func_kwargs

    @staticmethod
    def max(data):
        return torch.Tensor(np.nanmax(data, axis=3))

    @staticmethod
    def min(data):
        return torch.Tensor(np.nanmin(data, axis=3))

    @staticmethod
    def mean(data):
        return torch.Tensor(np.nanmean(data, axis=3))

    @staticmethod
    def var(data):
        return torch.Tensor(np.nanvar(data, axis=3))

    @staticmethod
    def count(data):
        """ Counts the number of non nan values. """
        return (1 - np.isnan(data)).float().sum(axis=3)

    @staticmethod
    def moments(data, n=2):
        """Gets statistical moments from the data.

        Args:
            data (torch.Tensor): Pytorch rolling window data.
            n (int): Moments to compute up to. Must be >=2 computes moments [2, 3, ..., n].
        """
        # Removes the mean of empty slice warning
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        assert n >= 2, "Number of moments is {}, must be >= 2.".format(n)

        # Pre computation
        nanmean = torch.Tensor(np.nanmean(data, axis=3)).unsqueeze(-1)
        frac = 1 / (data.size(3) - 1)
        mean_reduced = data - nanmean

        # Compute each moment individually
        moments = []
        for i in range(2, n+1):
            moment = frac * (mean_reduced ** i).sum(axis=3)
            moments.append(moment)
        moments = torch.cat(moments, dim=2)

        return moments

    def transform(self, data):
        # Remove mean of empty slice warning
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Error handling
        assert self.statistic in dir(self), 'Statistic {} is not implemented via this method.'

        # Setup function
        func = eval('self.{}'.format(self.statistic))

        # Make rolling
        rolling = pytorch_rolling(data, 1, self.window_length, self.step_size)

        # Apply and output
        output = func(rolling, **self.func_kwargs)

        return output

