from definitions import *
import numpy as np
from src.features.signatures.compute import RollingSignature, get_signature_feature_names


dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)

signatures = RollingSignature(window=3, depth=2, logsig=True).compute(dataset.loc[:, ['SBP', 'HR']])
cols = get_signature_feature_names(['SBP', 'HR'], depth=2, logsig=True)
dataset.add_features(signatures, cols)