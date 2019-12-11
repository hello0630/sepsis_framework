from definitions import *
import numpy as np
import torch
from sklearn.model_selection import cross_val_predict
from lightgbm import LGBMRegressor
from src.data.dicts import features as feature_dict
from src.data.transformers import ForwardFill, DerivedFeaturesTorch
from src.features.signatures.compute import RollingSignature
from src.features.signatures.augmentations import LeadLag, AddTime
from src.features.transfomers import RollingStatistic, FeaturePipeline
from src.models.model_selection import CustomStratifiedGroupKFold
from src.models.optimizers import ThresholdOptimizer, TorchThresholdOptimizer

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
    ('moments', feature_dict['non_demographic'], RollingStatistic(statistic='moments', window_length=7, func_kwargs={'n': 3})),
]
features = FeaturePipeline(steps=steps).transform(dataset)
dataset.add_features(features)

# Signature
# for feature in ['SOFA', 'MAP', 'BUN/CR']:
#     signatures = RollingSignature(window=7, depth=3, logsig=True).transform(AddTime().transform(dataset[feature]))
#     dataset.add_features(signatures)

# Extract machine learning data
X, y = dataset.to_ml()

# Setup cross val
cv_path = DATA_DIR + '/processed/cv/5_split.pickle'
if os.path.exists(cv_path):
    cv = load_pickle(cv_path)
else:
    cv = CustomStratifiedGroupKFold(n_splits=5).split(dataset)
    save_pickle(cv, cv_path)

# Classifier with parameters
clf = LGBMRegressor().set_params(**{'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6})

# Make predictions
predictions = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

# # Perform thresholding
print('Thresholding...')
scores = TorchThresholdOptimizer(dataset.labels, predictions).optimize_cv(cv, parallel=False)
print('Average: {:.3f}'.format(np.mean(scores)))

