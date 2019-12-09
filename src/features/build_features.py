from definitions import *
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from src.data.dicts import features
from src.features.signatures.compute import RollingSignature, get_signature_feature_names
from src.features.transfomers import RollingStatistic
from src.features.functions import pytorch_rolling

dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)


class Pipe():
    def __init__(self, steps):
        self.steps = steps

    def transform(self, dataset):
        outputs = []
        for name, cols, transformer in self.steps:
            output = transformer.transform(dataset[cols])
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-1)
        return outputs


steps = [
    ('count', features['laboratory'], RollingStatistic(statistic='count', window_length=8)),
    ('max', features['vitals'], RollingStatistic(statistic='max', window_length=6)),
    ('min', features['vitals'], RollingStatistic(statistic='min', window_length=6)),
    ('moments', features['non_demographic'], RollingStatistic(statistic='moments', window_length=8, func_kwargs={'n': 3})),
    ('signatures', ['HR', 'MAP'], RollingSignature(window=6, depth=3, logsig=True)),
]

outputs = Pipe(steps=steps).transform(dataset)

dataset.add_features(outputs, columns=None)

