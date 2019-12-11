import itertools
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.omni.decorators import timeit


class CustomStratifiedGroupKFold(BaseEstimator, TransformerMixin):
    """
    Picks folds with approx equal number of septic cases across folds.
    """
    def __init__(self, n_splits=5, seed=3):
        self.n_splits = n_splits
        self.seed = seed if isinstance(seed, int) else np.random.randint(10000)


    @timeit
    def split(self, dataset):
        # Set a seed so the same cv is used everytime
        np.random.seed(self.seed)

        # Find which ids contain a one
        id_labels = dataset.id_labels()
        contains_one = np.array([(labels.sum() > 0).item() for labels in id_labels])
        one_ids = dataset.ids[contains_one]

        # Shuffle and then get training and validation folds
        np.random.shuffle(one_ids)
        val_ones = np.array_split(one_ids, self.n_splits)
        train_ones = [list(set(one_ids) - set(x)) for x in val_ones]

        # Now get folds for the cases that did not develop the condition
        zero_ids = [x for x in dataset.ids if x not in one_ids]
        np.random.shuffle(zero_ids)
        val_zeros = np.array_split(np.array(zero_ids), self.n_splits)
        train_zeros = [list(set(zero_ids) - set(x)) for x in val_zeros]

        # Compile together
        id_groups = [(list(train_zeros[i]) + list(train_ones[i]), list(val_zeros[i]) + list(val_ones[i]))
                     for i in range(self.n_splits)]

        # Finally, get the indexes
        cv_iter = [(np.argwhere(np.isin(dataset.ids_long, x[0]) == True).reshape(-1),
                    np.argwhere(np.isin(dataset.ids_long, x[1]) == True).reshape(-1))
                   for x in id_groups]

        return cv_iter


if __name__ == '__main__':
    from definitions import *
    dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)

    CustomStratifiedGroupKFold().split(dataset)
