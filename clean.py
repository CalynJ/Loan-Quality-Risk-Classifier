import pandas as pd

train = pd.read_csv('/home/claude/converted/train-FIN_ANA_DATA_.csv')
test = pd.read_csv('/home/claude/converted/test-FIN_ANA_DATA_.csv')

print("BEFORE CLEANING")
print(train.isnull().sum()[train.isnull().sum() > 0])
print("CLIENT_TYPE junk rows:", (train['CLIENT_TYPE'] == '0').sum())

train = train[train['CLIENT_TYPE'] != '0']
train = train.dropna(subset=['INF_MARITAL_STATUS', 'INF_GENDER', 'COMPENSATION_CHARGED', 'CLIENT_TYPE'])
train['INSTALL_SIZE'] = train['INSTALL_SIZE'].fillna(0)
test['INSTALL_SIZE'] = test['INSTALL_SIZE'].fillna(0)

print("AFTER CLEANING")
print("Train shape:", train.shape)
print("Remaining nulls:", train.isnull().sum().sum())

train.to_csv('/home/claude/project/train_clean.csv', index=False)
test.to_csv('/home/claude/project/test_clean.csv', index=False)
print("Saved.")
