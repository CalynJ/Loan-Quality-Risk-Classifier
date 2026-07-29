import pandas as pd

# Load the raw data
train = pd.read_csv('train_clean.csv')

# 1. Remove junk placeholder values
# CLIENT_TYPE had 3 rows with "0" instead of a real category (Rural/Semi-urban/Urban)
# This is a data entry error, not a real category - so we drop those rows
train = train[train['CLIENT_TYPE'] != '0']

# 2. Drop rows with missing values in key categorical columns
# Only ~100 rows out of 37,408 were missing these - small enough to just remove
# rather than try to guess/impute a value
train = train.dropna(subset=['INF_MARITAL_STATUS', 'INF_GENDER', 
                               'COMPENSATION_CHARGED', 'CLIENT_TYPE'])

# 3. Fill missing INSTALL_SIZE with 0
# INSTALL_SIZE was mostly 0 already (75% of all rows had 0)
# so filling missing values with 0 matches the existing pattern in the data
# rather than introducing a fake average or guess
train['INSTALL_SIZE'] = train['INSTALL_SIZE'].fillna(0)

print("Final shape:", train.shape)
print("Remaining nulls:", train.isnull().sum().sum())
