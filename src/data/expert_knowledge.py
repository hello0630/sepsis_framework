"""
Script where we apply 'expert knowledge' to the data. That is, any domain expertise to generate useful features.
"""
from definitions import *
from src.features.transfomers import DerivedFeatures


if __name__ == '__main__':
    df = load_pickle(DATA_DIR + '/interim/from_raw/df.pickle')
    df = DerivedFeatures().transform(df)
    