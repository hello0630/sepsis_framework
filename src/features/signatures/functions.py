"""
functions.py
===================
Helper functions in signature computation.
"""
import torch
import signatory


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


def leadlag_slice(leadlag_data):
    """ Slice indexer to pull filter additional pieces from a leadlag transformation. """
    return slice(0, leadlag_data.size(1), 2)
