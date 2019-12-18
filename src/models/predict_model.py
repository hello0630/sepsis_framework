from definitions import *
import numpy as np
import torch
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBClassifier
from src.data.dataset.dataset import TimeSeriesDataset
from src.data.dicts import feature_dict, lgb_params
from src.features.signatures.compute import RollingSignature, DatasetSignatures
from src.features.signatures.augmentations import LeadLag, AddTime, PenOff, CumulativeSum
from src.features.transfomers import RollingStatistic, FeaturePipeline
from src.models.model_selection import CustomStratifiedGroupKFold
from src.models.optimizers import ThresholdOptimizer
# TODO Need to rethink some things, if we intend to do find min/max features then apply penoff, there is a lot of /
# TODO censored data. Instead we should aim to include time and allow for shorter intervals to be given, but the time
# TODO exist as a parameter, then the algo can decide for itself...

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
dataset = load_pickle(DATA_DIR + '/interim/preprocessed/dataset.dill')
# dataset.data.to(device)

# Compute some features
steps = [
    ('count', feature_dict['counts'], RollingStatistic(statistic='change', window_length=8)),
    ('max', feature_dict['vitals'], RollingStatistic(statistic='max', window_length=6)),
    ('min', feature_dict['vitals'], RollingStatistic(statistic='min', window_length=6)),
    # ('moments', feature_dict['non_demographic'], RollingStatistic(statistic='moments', window_length=8, func_kwargs={'n': 3})),
]
features = FeaturePipeline(steps=steps).transform(dataset)
names = [x + '_count' for x in feature_dict['counts']] + [x + '_max' for x in feature_dict['vitals']] + [x + '_min' for x in feature_dict['vitals']]
dataset.add_features(features, names)


# Extract machine learning data
X, _ = dataset.to_ml()
y = dataset.labels_utility

# Setup cross val
cv_path = DATA_DIR + '/processed/cv/5_split.pickle'
if all([os.path.exists(cv_path), True]):
    cv = load_pickle(cv_path)
else:
    cv = CustomStratifiedGroupKFold(n_splits=5, seed=8).split(dataset)
    save_pickle(cv, cv_path)

# Regressor
params = load_pickle(MODELS_DIR + '/parameters/lgb/random_grid_fullds.pickle')
clf = LGBMRegressor().set_params(**lgb_params).set_params(**{'min_child_samples': 199, 'min_child_weight': 38, 'num_leaves': 49})
predictions = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

# Perform thresholding
print('Thresholding...')
scores = ThresholdOptimizer(dataset.labels.numpy(), predictions).optimize_cv(cv, parallel=True)
print('Average: {:.3f}'.format(np.mean(scores)))

