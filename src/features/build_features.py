from definitions import *
import numpy as np
import torch
from src.data.dicts import features
from src.features.signatures.compute import RollingSignature, get_signature_feature_names
from src.features.functions import pytorch_rolling

dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)



data = dataset.loc[:, features['laboratory']]
RollingStatistic(statistic='max', window_length=6, step_size=1).transform(data)

# Compute number of measurements
unfold = pytorch_rolling(data, 1, 8, 1)
num_measurements = (1 - np.isnan(unfold)).float().sum(axis=3)
col_names = ['COUNT.LB8_' + x for x in features['laboratory']]
dataset.add_features(num_measurements, col_names)

# Compute max val of the vitals
data = dataset.loc[:, features['vitals']]
unfold = pytorch_rolling(data, 1, 6, 1).numpy()
rolling_max = torch.Tensor(np.nanmax(unfold, axis=3))
max_cols = ['MAX.LB6_' + x for x in features['vitals']]
dataset.add_features(rolling_max, max_cols)

# Now some signatures
sig_data = dataset.loc[:, ['SBP', 'HR']]
signatures = RollingSignature(window=6, depth=3, logsig=True).compute(sig_data)
sig_names = get_signature_feature_names(['SBP', 'HR'], 3, True, append_string='SIG.')
dataset.add_features(signatures, sig_names)



