"""
compute.py
================================
Classes for signature computation.
"""
from definitions import *
import numpy as np
import torch
import signatory


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


def get_signature_feature_names(feature_names, depth, logsig=False, append_string=None):
    """Given some input feature names, gets the corresponding signature features names up to a given depth.

    Args:
        feature_names (list): A list of feature names that will correspond to the features of some input path to the
                              signature transformation.
        depth (int): The depth of the signature computed to.
        logsig (bool): True for names in the logsig transform.
        append_string (str): A string to append to the start of each col name. This is used to signify what computation
                             was performed in signature generation to help distinguish from original column names.

    Returns:
        list: List of feature names that correspond to the output columns of the signature transformation.
    """
    channels = len(feature_names)
    if not logsig:
        words = signatory.all_words(channels, depth)
        words_lst = [list(x) for x in words]
        sig_names = ['|'.join([feature_names[x] for x in y]) for y in words_lst]
    else:
        lyndon = signatory.lyndon_brackets(channels, depth)
        lyndon_str = [str(l) for l in lyndon]
        for num in list(range(len(feature_names))[::-1]):
            for i in range(len(lyndon_str)):
                lyndon_str[i] = lyndon_str[i].replace(str(num), str(feature_names[num]))
        sig_names = lyndon_str
    if append_string != None:
        sig_names = [append_string + '_' + x for x in sig_names]
    return sig_names


if __name__ == '__main__':
    from definitions import *
    from sklearn.pipeline import Pipeline
    from src.features.signatures.augmentations import *
    from src.features.transfomers import RollingStatistic
    dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)

    data = torch.tensor([1.0, 2.0, 4., 5.]).reshape(1, 4, 1)

    rs = RollingStatistic(statistic='moments', window_length=3, func_kwargs={'n': 3}).transform(data)

    # cs_steps = [
    #     ('cumulative_sum', CumulativeSum(append_zero=True)),
    #     ('lead_lag', LeadLag()),
    # ]
    # cs_pipe = Pipeline(cs_steps)
    # out = cs_pipe.transform(data)
    # signatures = signatory.logsignature(out, 2)
    # data.var()
    #
