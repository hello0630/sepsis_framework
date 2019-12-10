from definitions import *
import multiprocessing
import torch
from nevergrad.optimization import optimizerlib
from nevergrad import instrumentation as inst


class TorchThresholdOptimizer():
    def __init__(self, labels, predictions, budget=200, num_workers=1):
        self.labels = labels
        self.predictions = predictions
        self.scores = torch.Tensor(load_pickle(DATA_DIR + '/processed/labels/full_scores.pickle').values)

        # Nevergrad options
        self.budget = budget
        self.num_workers = num_workers

    def score(self, thresh, idx=None):
        # Get the corresponding indexes
        predictions = self.predictions[idx] if idx is not None else self.predictions
        scores = self.scores[idx] if idx is not None else self.scores

        # Precompute the inaction and perfect scores
        inaction_score = scores[:, 0].sum()
        perfect_score = scores[:, [0, 1]].max(axis=1)[0].sum()

        # Apply the threshold
        predictions = (predictions > thresh).astype(int)

        # Get the actual score
        actual_score = scores[:, 1][predictions == 1].sum() + scores[:, 0][predictions == 0].sum()

        # Get the normalized score
        normalized_score = (actual_score - inaction_score) / (perfect_score - inaction_score)

        return normalized_score.item()

    def optimize(self, idx):
        # Set optimizer and instrumentation (bounds)
        instrum = inst.Instrumentation(*[inst.var.Array(1).asscalar().bounded(-0.2, 0.2)])
        optimizer = optimizerlib.TwoPointsDE(instrumentation=instrum, budget=self.budget, num_workers=self.num_workers)

        # Optimize
        recommendation = optimizer.optimize(
            lambda thresh: -self.score(thresh, idx=idx)
        )

        # Get the threshold and return the score
        threshold = recommendation.args[0]

        return threshold

    def _optimize_cv_func(self, train_idx, test_idx):
        # Optimize on the training set.
        threshold = self.optimize(train_idx)

        # Apply the threshold to the test set
        test_score = self.score(threshold, test_idx)
        print('\tScore on cv fold: {:.3f}'.format(test_score))

        return test_score

    def optimize_cv(self, cv, parallel=False):
        """ Optimizes the threshold over each CV fold. """
        scores = parallel_cv_loop(self._optimize_cv_func, cv, parallel=parallel)
        return scores



class ThresholdOptimizer():
    """
    Given labels and proba or regression predictions, finds the optimal threshold to attain the max utility score.
    """
    def __init__(self, labels=None, preds=None, budget=200, parallel=False, cv_num=False, jupyter=False):
        self.labels = labels
        self.preds = preds
        self.budget = budget
        self.num_workers = 1 if parallel is False else multiprocessing.cpu_count()
        self.cv_num = cv_num

        # Get the scores
        self.scores_loc = DATA_DIR + '/processed/labels/full_scores.pickle' if jupyter is False else ROOT_DIR + '/data/processed/labels/full_scores.pickle'
        self.scores = load_pickle(self.scores_loc)

    @staticmethod
    def score_func(scores, predictions, inaction_score, perfect_score, thresh=0):
        """ The utility score function, scores and predictions must be entered as numpy arrays. """
        # Apply the threshold
        predictions = (predictions > thresh).astype(int)

        # Get the actual score
        actual_score = scores[:, 1][predictions == 1].sum() + scores[:, 0][predictions == 0].sum()

        # Get the normalized score
        normalized_score = (actual_score - inaction_score) / (perfect_score - inaction_score)

        return normalized_score

    def optimize_utility(self, labels, predictions):
        """ Main function for optimization of a threshold given labels and predictions. """
        if isinstance(predictions, pd.Series):
            predictions = predictions.values

        # We only want scores correspondent with labels
        scores = self.scores.loc[labels.index].values

        # Give bounds
        instrum = inst.Instrumentation(*[inst.var.Array(1).asscalar().bounded(-0.2, 0.2)])

        # Set optimizer
        optimizer = optimizerlib.TwoPointsDE(instrumentation=instrum, budget=self.budget, num_workers=self.num_workers)

        # Precompute the inaction and perfect scores
        inaction_score = scores[:, 0].sum()
        perfect_score = scores[:, [0, 1]].max(axis=1).sum()

        # Optimize
        recommendation = optimizer.optimize(
            lambda thresh: -self.score_func(scores, predictions, inaction_score, perfect_score, thresh=thresh)
        )

        # Get the threshold and return the score
        threshold = recommendation.args[0]
        score = self.score_func(scores, predictions, inaction_score, perfect_score, thresh=threshold)

        return threshold, score

    def cv_func(self, train_idx, test_idx, cv_num=False):
        """ The function run for each fold of the cross_val_threshold method. """
        # Split test train
        labels_train, labels_test = self.labels.iloc[train_idx], self.labels.iloc[test_idx]
        preds_train, preds_test = self.preds.iloc[train_idx], self.preds.iloc[test_idx]

        # Optimise the training
        threshold, score = self.optimize_utility(labels_train, preds_train)

        # Apply to the testing
        preds_test_thresh = (preds_test >= threshold).astype(int)
        test_score = ComputeNormalizedUtility().score(labels_test, preds_test_thresh, cv_num=cv_num)
        ppprint('\tScore on cv fold: {:.3f}'.format(test_score))

        return preds_test_thresh, test_score, threshold

    @timeit
    def cross_val_threshold(self, cv, give_cv_num=False, parallel=True):
        """
        Similar to cross val predict, performs the thresholding algorithm on the given cv folds.

        Note that if this is specified, labels and pred must be preloaded in __init__()
        """
        results = parallel_cv_loop(self.cv_func, cv, give_cv_num=give_cv_num, parallel=parallel)

        # Open out results
        preds = pd.concat([x[0] for x in results], axis=0)
        scores = [x[1] for x in results]
        thresholds = [x[2] for x in results]

        return preds, scores, thresholds


def parallel_cv_loop(func, cv, parallel=True):
    """
    Performs a parallel training loop over the cv train_idx and test_idxs.

    Example:
        - func will usually be a class that contains df, labels info but __call__ method will run a single training loop
        given train_idx, test_idx
        - This will run func.__call__(train_idx, test_idx) for each idx pair in cv and return results

    Args:
        func (object): Class that has information relating to data, labels and takes a __call__(train_idx, test_idx) to
                       run loop.
        cv (list): List of [(train_idx, test_idx), ...] pairs.
        give_cv_num (bool): Gives the cv num to the underlying function, used when using the full dataset and loading
                            precomputed arrays for a specific cv_num
        parallel (bool): Set to false for a for loop (allows for debugging)

    Return:
        (list): A list of whatever func outputs for each cv idxs.
    """
    if parallel:
        pool = Pool(len(cv))
        results = pool.starmap(
            func, cv
        )
        pool.close()
    else:
        results = []
        for args in cv:
            results.append(func(*args))

    return results
