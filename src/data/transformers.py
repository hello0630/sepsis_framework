from definitions import *
import numpy as np
from src.external.evaluate_sepsis_score import compute_prediction_utility


class LabelsToScores(BaseIDTransformer):
    """ Given a set of 0-1 labels, transforms to find the score for predicting a 1 or a 0

    """
    def transform_id(self, labels):
        if isinstance(labels, pd.DataFrame):
            labels = labels['SepsisLabel']

        # Get same length zeros and ones
        zeros = np.zeros(shape=(len(labels)))
        ones = np.ones(shape=(len(labels)))

        # Get scores for predicting zero or 1
        zeros_pred = compute_prediction_utility(labels.values, zeros, return_all_scores=True)
        ones_pred = compute_prediction_utility(labels.values, ones, return_all_scores=True)

        # Input scores of 0 and 1
        scores = np.concatenate([zeros_pred.reshape(-1, 1), ones_pred.reshape(-1, 1)], axis=1)
        scores = pd.DataFrame(index=labels.index, data=scores, columns=[0, 1])

        # Make an overall utilty score equal to one_score - zero_score which encodes the benefit of the 1 prediction
        scores['utility'] = scores[1] - scores[0]

        return scores
