from definitions import *
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DerivedFeatures(TransformerMixin, BaseEstimator):
    """
    Adds any derived features thought to be useful
        - Shock Index: HR/SBP
        - Bun/Creatinine ratio: Bun/Creatinine
        - Hepatic SOFA: Bilirubin SOFA score

    # Can add renal and neruologic sofa
    """
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    @staticmethod
    def hepatic_sofa(df):
        """ Updates a hepatic sofa score """
        hepatic = np.zeros(shape=df.shape[0])

        # Bili
        bilirubin = df['Bilirubin_total'].values
        hepatic[bilirubin < 1.2] += 0
        hepatic[(bilirubin >= 1.2) & (bilirubin < 1.9)] += 1
        hepatic[(df['Bilirubin_total'] >= 1.9) & (bilirubin < 5.9)] += 2
        hepatic[(bilirubin >= 5.9) & (bilirubin < 11.9)] += 3
        hepatic[(bilirubin >= 11.9)] += 4

        # MAP
        hepatic[df['MAP'].values < 70] += 1

        # Creatinine
        creatinine = df['Creatinine'].values
        hepatic[(creatinine >= 1.2) & (creatinine < 1.9)] += 1
        hepatic[(creatinine >= 1.9) & (creatinine < 3.4)] += 2
        hepatic[(creatinine >= 3.5) & (creatinine < 4.9)] += 3
        hepatic[(creatinine >= 4.9)] += 4

        # Platelets
        platelets = df['Platelets'].values
        hepatic[(platelets >= 100) & (platelets < 150)] += 1
        hepatic[(platelets >= 50) & (platelets < 100)] += 2
        hepatic[(platelets >= 20) & (platelets < 49)] += 3
        hepatic[(platelets < 20)] += 4

        return hepatic

    @staticmethod
    def sirs_criteria(df):
        # Create a dataframe that stores true false for each category
        df_sirs = pd.DataFrame(index=df.index, columns=['temp', 'hr', 'rr.paco2', 'wbc'])
        df_sirs['temp'] = ((df['Temp'] > 38) | (df['Temp'] < 36))
        df_sirs['hr'] = df['HR'] > 90
        df_sirs['rr.paco2'] = ((df['Resp'] > 20) | (df['PaCO2'] < 32))
        df_sirs['wbc'] = ((df['WBC'] < 4) | (df['WBC'] > 12))

        # Sum each row, if >= 2 then mar as SIRS
        sirs = pd.to_numeric((df_sirs.sum(axis=1) >= 2) * 1)

        # Leave the binary and the path sirs
        sirs_df = pd.concat([sirs, df_sirs.sum(axis=1)], axis=1)
        sirs_df.columns = ['SIRS', 'SIRS_path']

        return sirs_df

    @staticmethod
    def mews_score(df):
        mews = np.zeros(shape=df.shape[0])

        # SBP
        sbp = df['SBP'].values
        mews[sbp <= 70] += 3
        mews[(70 < sbp) & (sbp <= 80)] += 2
        mews[(80 < sbp) & (sbp <= 100)] += 1
        mews[sbp >= 200] += 2

        # HR
        hr = df['HR'].values
        mews[hr < 40] += 2
        mews[(40 < hr) & (hr <= 50)] += 1
        mews[(100 < hr) & (hr <= 110)] += 1
        mews[(110 < hr) & (hr < 130)] += 2
        mews[hr >= 130] += 3

        # Resp
        resp = df['Resp'].values
        mews[resp < 9] += 2
        mews[(15 < resp) & (resp <= 20)] += 1
        mews[(20 < resp) & (resp < 30)] += 2
        mews[resp >= 30] += 3

        return mews

    @staticmethod
    def qSOFA(df):
        qsofa = np.zeros(shape=df.shape[0])
        qsofa[df['Resp'].values >= 22] += 1
        qsofa[df['SBP'].values <= 100] += 1
        return qsofa

    @staticmethod
    def SOFA(df):
        sofa = np.zeros(shape=df.shape[0])

        # Coagulation
        platelets = df['Platelets'].values
        sofa[platelets >= 150] += 0
        sofa[(100 <= platelets) & (platelets < 150)] += 1
        sofa[(50 <= platelets) & (platelets < 100)] += 2
        sofa[(20 <= platelets) & (platelets < 50)] += 3
        sofa[platelets < 20] += 4

        # Liver
        bilirubin = df['Bilirubin_total'].values
        sofa[bilirubin < 1.2] += 0
        sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
        sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
        sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
        sofa[bilirubin > 11.9] += 4

        # Cardiovascular
        map = df['MAP'].values
        sofa[map >= 70] += 0
        sofa[map < 70] += 1

        # Creatinine
        creatinine = df['Creatinine'].values
        sofa[creatinine < 1.2] += 0
        sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
        sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
        sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
        sofa[creatinine > 4.9] += 4

        return sofa

    @staticmethod
    def SOFA_max_24(s):
        """ Get the max value of the SOFA score over the prev 24 hrs """
        def find_24_hr_max(s):
            prev_24_hrs = pd.concat([s.shift(i) for i in range(24)], axis=1).values[:, ::-1]
            return pd.Series(index=s.index, data=np.nanmax(prev_24_hrs, axis=1))
        sofa_24 = s.groupby('id').apply(find_24_hr_max)
        return sofa_24

    @staticmethod
    def SOFA_deterioration_new(s):
        def check_24hr_deterioration(s):
            """ Check the max deterioration over the last 24 hours, if >= 2 then mark as a 1"""
            prev_24_hrs = pd.concat([s.shift(i) for i in range(24)], axis=1).values[:, ::-1]

            def max_deteriorate(arr):
                return np.nanmin([arr[i] - np.nanmax(arr[i+1:]) for i in range(arr.shape[-1]-1)])

            tfr_hr_min = np.apply_along_axis(max_deteriorate, 1, prev_24_hrs)
            return pd.Series(index=s.index, data=tfr_hr_min)
        sofa_det = s.groupby('id').apply(check_24hr_deterioration)
        return sofa_det

    @staticmethod
    def SOFA_deterioration(s):
        def check_24hr_deterioration(s):
            """ Check the max deterioration over the last 24 hours, if >= 2 then mark as a 1"""
            prev_23_hrs = pd.concat([s.shift(i) for i in range(1, 24)], axis=1).values
            tfr_hr_min = np.nanmin(prev_23_hrs, axis=1)
            return pd.Series(index=s.index, data=(s.values - tfr_hr_min))
        sofa_det = s.groupby('id').apply(check_24hr_deterioration)
        sofa_det[sofa_det < 0] = 0
        sofa_det = sofa_det
        return sofa_det

    @staticmethod
    def septic_shock(df):
        shock = np.zeros(shape=df.shape[0])
        shock[df['MAP'].values < 65] += 1
        shock[df['Lactate'].values < 2] += 1
        return shock

    def transform(self, df):
        # Compute things
        df['ShockIndex'] = df['HR'].values / df['SBP'].values
        df['ShockIndex_AgeNorm'] = df['HR'].values / (df['SBP'].values * df['Age'].values)
        df['BUN/CR'] = df['BUN'].values / df['Creatinine'].values
        df['SaO2/FiO2'] = df['SaO2'].values / df['FiO2'].values

        # SOFA
        df['SOFA'] = self.SOFA(df[['Platelets', 'MAP', 'Creatinine', 'Bilirubin_total']])
        # df['SOFA_deterioration'] = self.SOFA_deterioration(df['SOFA'])
        # df['SOFA_max_24hrs'] = self.SOFA_max_24(df['SOFA'])
        # df['HepaticSOFA'] = self.hepatic_sofa(df)
        # df['qSOFA'] = self.qSOFA(df)
        # df['SOFA_24hrmaxdet'] = self.SOFA_deterioration(df['SOFA_max_24hrs'])
        # df['SOFA_deterioration_new'] = self.SOFA_deterioration_new(df['SOFA_max_24hrs'])
        # df['SepticShock'] = self.septic_shock(df)

        # Other scores
        # sirs_df = self.sirs_criteria(df)
        # df['MEWS'] = self.mews_score(df)
        # df['SIRS'] = sirs_df['SIRS']
        # df['SIRS_path'] = sirs_df['SIRS_path']
        return df


if __name__ == '__main__':
    dataset = load_pickle(DATA_DIR + '/interim/from_raw/sepsis_dataset.dill', use_dill=True)
