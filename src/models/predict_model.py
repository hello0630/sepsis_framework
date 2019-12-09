from definitions import *
import numpy as np
import torch
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMRegressor
from src.data.dicts import features
from src.data.transformers import ForwardFill, DerivedFeaturesTorch
from src.features.signatures.compute import RollingSignature, get_signature_feature_names
from src.features.transfomers import RollingStatistic, FeaturePipeline
from src.models.model_selection import CustomStratifiedGroupKFold


# Load the data
dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)

# Perform forward fill
ffill = ForwardFill().transform(dataset.data)
dataset.update_data(ffill)

# Derive features
dataset = DerivedFeaturesTorch().transform(dataset)

# Compute some features
steps = [
    ('count', features['laboratory'], RollingStatistic(statistic='count', window_length=8)),
    ('max', features['vitals'], RollingStatistic(statistic='max', window_length=6)),
    ('min', features['vitals'], RollingStatistic(statistic='min', window_length=6)),
    ('moments', features['non_demographic'], RollingStatistic(statistic='moments', window_length=8, func_kwargs={'n': 3})),
    ('signatures', ['HR', 'MAP'], RollingSignature(window=6, depth=3, logsig=True)),
]
features = FeaturePipeline(steps=steps).transform(dataset)
dataset.add_features(features)

# Extract machine learning data
X, y = dataset.to_ml()

# Setup cross val
cv = CustomStratifiedGroupKFold(n_splits=5).split(dataset)

# Classifier with parameters
clf = LGBMRegressor().set_params(**{'n_estimators': 100, 'learning_rate': 0.1})

# Make predictions
predictions = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

