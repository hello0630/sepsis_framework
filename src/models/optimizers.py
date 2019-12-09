from definitions import *
import multiprocessing
from nevergrad.optimization import optimizerlib
from nevergrad import instrumentation as inst


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
