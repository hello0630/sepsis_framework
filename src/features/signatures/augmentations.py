"""
augmentations.py
================================
Contains sklearn transformers for path augmentations to be applied before computation of signatures.
"""
import torch
from sklearn.base import BaseEstimator, TransformerMixin


class AddTime(BaseEstimator, TransformerMixin):
    """Add time component to each path.

    For a path of shape [N, L, C] this adds a time channel to be placed at the first index. The time channel will be of
    length L and scaled to exist in [0, 1].
    """
    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        # Batch and length dim
        B, L = data.shape[0], data.shape[1]

        # Time scaled to 0, 1
        time_scaled = torch.linspace(0, 1, L).repeat(B, 1).view(B, L, 1)

        return torch.cat((time_scaled, data), 2)


class PenOff(BaseEstimator, TransformerMixin):
    """Adds a 'penoff' dimension to each path. """
    def fit(self, data, labels=None):
        return self

    def transform(self, X):
        # Batch, length, channels
        B, L, C = X.shape[0], X.shape[1], X.shape[2]

        # Add in a dimension of ones
        X_pendim = torch.cat((torch.ones(B, L, 1), X), 2)

        # Add pen down to 0
        pen_down = X_pendim[:, [-1], :]
        pen_down[:, :, 0] = 0
        X_pendown = torch.cat((X_pendim, pen_down), 1)

        # Add home
        home = torch.zeros(B, 1, C + 1)
        X_penoff = torch.cat((X_pendown, home), 1)

        return X_penoff


class LeadLag(BaseEstimator, TransformerMixin):
    """Applies the leadlag transformation to each path.

    Example:
        This is a string man
            [1, 2, 3] -> [[1, 1], [2, 1], [2, 2], [3, 2], [3, 3]]
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Interleave
        X_repeat = X.repeat_interleave(2, dim=1)

        # Split out lead and lag
        lead = X_repeat[:, 1:, :]
        lag = X_repeat[:, :-1, :]

        # Combine
        X_leadlag = torch.cat((lead, lag), 2)

        return X_leadlag


class ShiftToZero():
    """Performs a translation so all paths begin at zero. """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X - X[:, [0], :]


class CumulativeSum():
    """Cumulative sum transform. """

    def __init__(self, append_zero=False):
        self.append_zero = append_zero

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.append_zero:
            X = AppendZero().transform(X)
        return torch.cumsum(X, 1)


