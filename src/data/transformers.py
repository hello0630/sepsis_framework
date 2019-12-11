from definitions import *
import numpy as np
import torch
from src.external.evaluate_sepsis_score import compute_prediction_utility


class LabelsToScores(BaseIDTransformer):
    """ Given a set of 0-1 labels, transforms to find the score for predicting a 1 or a 0.

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


class DerivedFeaturesTorch():
    """ Various features pre-derived from the data that are computed from the sepsis dataset class. """
    @staticmethod
    def sofa(dataset):
        N, L, C = dataset.data.size()
        sofa = torch.zeros(N, L, 1)

        # Coagulation
        platelets = dataset['Platelets']
        sofa[platelets >= 150] += 0
        sofa[(100 <= platelets) & (platelets < 150)] += 1
        sofa[(50 <= platelets) & (platelets < 100)] += 2
        sofa[(20 <= platelets) & (platelets < 50)] += 3
        sofa[platelets < 20] += 4

        # Liver
        bilirubin = dataset['Bilirubin_total']
        sofa[bilirubin < 1.2] += 0
        sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
        sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
        sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
        sofa[bilirubin > 11.9] += 4

        # Cardiovascular
        map = dataset['MAP']
        sofa[map >= 70] += 0
        sofa[map < 70] += 1

        # Creatinine
        creatinine = dataset['Creatinine']
        sofa[creatinine < 1.2] += 0
        sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
        sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
        sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
        sofa[creatinine > 4.9] += 4

        return sofa

    @timeit
    def transform(self, dataset):
        # SOFA
        dataset.add_features(self.sofa(dataset), ['SOFA'])

        # Shock Index
        shock_index = dataset['HR'] / (dataset['SBP'] * dataset['Age'])
        dataset.add_features(shock_index, ['ShockIndexAgeNorm'])

        # BUN/CR
        bun_cr = dataset['BUN'] / dataset['Creatinine']
        dataset.add_features(bun_cr, ['BUN/CR'])

        # SaO2/FiO2
        sao2_fio2 = dataset['SaO2'] / dataset['FiO2']
        dataset.add_features(sao2_fio2, ['SaO2/FiO2'])

        return dataset


class ForwardFill():
    """ Forward fill for a torch tensor.

    This (currently) assumes a torch tensor input of shape [N, L, C] and will forward will along the 2nd (L)
    dimension.

    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    @staticmethod
    def ffill2d(arr):
        """ 2d ffill. """
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
        return out

    @timeit
    def transform(self, data):
        data_ffilled = torch.Tensor([self.ffill2d(x.numpy().T) for x in data]).transpose(1, 2)
        return data_ffilled



