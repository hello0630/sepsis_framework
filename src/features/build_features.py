from definitions import *
import numpy as np
import torch
from src.features.signatures.compute import RollingSignature, get_signature_feature_names


dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)

signatures = RollingSignature(window=3, depth=2, logsig=True).compute(dataset.loc[:, ['SBP', 'HR']])
cols = get_signature_feature_names(['SBP', 'HR'], depth=2, logsig=True, append_string='S1')
dataset.add_features(signatures, cols)

dataset.data.shape
len(dataset.columns)

df = dataset.to_frame()