"""
This file transforms the raw .psv files into a dataframe.
"""
from definitions import *
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.transformers import LabelsToScores


def load_to_dataframe():
    # File locations
    locations = [DATA_DIR + '/raw/' + x for x in ['training_A', 'training_B']]

    # Ready to store and concat
    data = []

    # Make dataframe with
    id = 0
    hospital = 1
    for loc in tqdm(locations):
        for file in tqdm(os.listdir(loc)):
            id_df = pd.read_csv(loc + '/' + file, sep='|')
            id_df['id'] = id    # Give a unique id
            id_df['hospital'] = hospital    # Note the hospital
            data.append(id_df)
            id += 1
        hospital += 1

    # Concat for df
    df = pd.concat(data)

    # Sort index and reorder columns
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)
    df = df[['id', 'time'] + [x for x in df.columns if x not in ['id', 'time', 'SepsisLabel']] + ['SepsisLabel']]

    return df


if __name__ == '__main__':
    df = load_to_dataframe()

    # Save frame
    save_pickle(df, DATA_DIR + '/interim/from_raw/df.pickle')

    # Labels -> scores
    scores = LabelsToScores().transform(df)
    save_pickle(scores['utility'], DATA_DIR + '/processed/labels/utility_scores.pickle')
    save_pickle(scores, DATA_DIR + '/processed/labels/full_scores.pickle')
    save_pickle(df['SepsisLabel'], DATA_DIR + '/processed/labels/binary.pickle')



