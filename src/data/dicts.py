"""
A collection of dictionaries containing lists and information to be accessed elsewhere.
"""

feature_dict = {
    'vitals': ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2'],
    'laboratory': ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
                   'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium',
                   'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                   'Fibrinogen', 'Platelets'],
    'demographics': ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'hospital']
}

feature_dict['non_demographic'] = feature_dict['vitals'] + feature_dict['laboratory']
feature_dict['counts'] = ['Temp'] + feature_dict['laboratory']
feature_dict['counts_'] = ['Temp_count'] + [x + '_count' for x in feature_dict['laboratory']]


lgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'colsample_bytree': 0.5884502388447466,
    'min_child_samples': 191,
    'min_child_weight': 32,
    'min_split_gain': 0.0,
    'subsample': 0.477870931,
    'subsample_for_bin': 200000,
    'num_leaves': 36,
    'reg_alpha': 100,
    'reg_lambda': 20,
    'n_jobs': -1,
    # 'random_state': 1,
}
