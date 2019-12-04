"""
Script where we apply 'expert knowledge' to the data. That is, any domain expertise to generate useful features.
"""
from definitions import *
from src.features.transfomers import DerivedFeatures


if __name__ == '__main__':
    dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.pickle')
    df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')
    unfold = dataset.data.unfold(1, 5, 1)
    m, _ = unfold.max(axis=3)
    m[dataset.ids == 0][:, :, dataset.columns == 'HR']
