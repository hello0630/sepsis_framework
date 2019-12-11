from definitions import *
import numpy as np
import torch
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMRegressor
from src.data.dicts import feature_dict, lgb_params
from src.features.signatures.compute import RollingSignature, DatasetSignatures
from src.features.signatures.augmentations import LeadLag, AddTime
from src.features.transfomers import RollingStatistic, FeaturePipeline
from src.models.model_selection import CustomStratifiedGroupKFold
from src.models.optimizers import ThresholdOptimizer

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
dataset = load_pickle(DATA_DIR + '/interim/preprocessed/dataset.dill', use_dill=True)
dataset.data.to(device)

# Compute some features
steps = [
    ('count', feature_dict['counts'], RollingStatistic(statistic='count', window_length=8)),
    ('max', feature_dict['vitals'], RollingStatistic(statistic='max', window_length=6)),
    ('min', feature_dict['vitals'], RollingStatistic(statistic='min', window_length=6)),
    # ('moments', feature_dict['non_demographic'], RollingStatistic(statistic='moments', window_length=7, func_kwargs={'n': 3})),
]
features = FeaturePipeline(steps=steps).transform(dataset)
dataset.add_features(features)

# Change counts to count/time
# dataset[[x for x in dataset.columns if '_count' in str(x)]] /= dataset[[x for x in dataset.columns if '_count' in str(x)]] / dataset['ICULOS']

# Signatures
augmentations = [
    # AddTime(),
    LeadLag(),
]
ds = DatasetSignatures(augmentations, window=14, depth=3, logsig=True, nanfill=True)
features = ['SOFA', 'MAP', 'BUN/CR', 'HR', 'SBP', 'ShockIndexAgeNorm']
# features = ['SOFA', 'MAP', 'BUN/CR', 'HR', 'SBP']
signatures = ds.transform(dataset, features)
print(dataset.data.shape)
dataset.add_features(signatures)
print(dataset.data.shape)

# Extract machine learning data
X, y = dataset.to_ml()

# Setup cross val
cv = load_pickle(DATA_DIR + '/processed/cv/5_split.pickle')

# Classifier with parameters
clf = LGBMRegressor().set_params(**lgb_params)

# Make predictions
predictions = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

# # Perform thresholding
print('Thresholding...')
scores = ThresholdOptimizer(dataset.labels, predictions).optimize_cv(cv, parallel=False)
print('Average: {:.3f}'.format(np.mean(scores)))

