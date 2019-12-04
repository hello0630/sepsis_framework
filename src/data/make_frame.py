"""
This file transforms the raw .psv files into a dataframe.
"""
from definitions import *
import pandas as pd
from src.data.transformers import LabelsToScores




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