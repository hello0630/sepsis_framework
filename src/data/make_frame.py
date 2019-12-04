"""
This file transforms the raw .psv files into a dataframe.
"""
from definitions import *
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from src.data.transformers import LabelsToScores


class SepsisDataset(Dataset):
    def __init__(self, data=None, labels=None, columns=None, lengths=None, ids=None):
        self.data = data
        self.labels = labels
        self.columns = columns
        self.lengths = lengths
        self.ids = ids

        # Additional indexing methods
    def from_dataframe(self, df):
        # Get id and labels
        ids = df['id']
        self.ids = ids.unique()
        self.labels = df['SepsisLabel']

        # Drop non-needed cols
        df = df.drop(['id', 'time', 'SepsisLabel'], axis=1)
        self.columns = df.columns

        # Pad and store
        paths = [torch.Tensor(df[ids == id]).values for id in self.ids]
        self.data = torch.nn.utils.rnn.pad_sequence(paths, padding_value=np.nan)
        self.lengths = [len(p) for p in paths]

        return self



if __name__ == '__main__':
    # File locations
    locations = [DATA_DIR + '/raw/' + x for x in ['training_A', 'training_B']]

    # Ready to store and concat
    data = []

    # Make dataframe with
    id = 0
    hospital = 1
    for loc in locations:
        for file in os.listdir(loc):
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

    # Save frame
    save_pickle(df, DATA_DIR + '/interim/from_raw/df.pickle')

    # Labels -> scores
    scores = LabelsToScores().transform(df)
    save_pickle(scores['utility'], DATA_DIR + '/processed/labels/utility_scores.pickle')
    save_pickle(scores, DATA_DIR + '/processed/labels/full_scores.pickle')



