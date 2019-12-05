from definitions import *
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.functions import pytorch_rolling


class RollingStatistic():
    """Applies statistics to a rolling window along the time dimension of a tensor.

    Given an input tensor of shape [N, L, C] and a specified window size, W, this function first expands the tensor to
    one of shape [N, L, C, W] where W has expanded out the time dimension (this here is the L-dimension). The final
    dimension contains the most recent W time-steps (with nans if not filled). The specified statistic is then computed
    along this W dimension to give the statistic over the rolling window.
    """
    def __init__(self, statistic, window_length, step_size=1):
        self.statistic = statistic
        self.window_length = window_length
        self.step_size = step_size

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

    def transform(self, data):
        # Error handling
        assert self.statistic in dir(self), 'Statistic {} is not implemented via this method.'

        # Setup function
        func = eval('self.{}'.format(self.statistic))

        # Make rolling
        rolling = pytorch_rolling(data, 1, 8, 1)

        # Apply and output
        output = func(rolling)

        return output

