# Package constants
# -----------------
MODELS_PARAMS = {'MODELS_PATH': "../models",
                 'TEST_1REG_CP_ALPHA': 0.1,
                 'TEST_PRECLF_FILE': "pre_classifier.joblib",
                 'TEST_3REG_FILES_LOW': "pd_regressor_5.1_low.joblib",
                 'TEST_3REG_FILES_MID': "pd_regressor_5.1_mid.joblib",
                 'TEST_3REG_FILES_HIGH': "pd_regressor_5.1_high.joblib"}
DAYS_INTERP = 365


# Testing constants
# -----------------
TESTING_DELTAS = dict(field_126={'2017': 22, '2023': 11, '2018': 0, '2020': 9, '2021': 14, '2019': 43, '2022': 3},
                      field_148={'2017': 20, '2023': 13, '2018': 0, '2020': 2, '2021': 39, '2019': 34, '2022': 2})
