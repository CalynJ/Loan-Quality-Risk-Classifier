import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# ---------------------------------------------------------
# 1. LOAD DATA & DEFINE TARGET
# ---------------------------------------------------------
df = pd.read_csv('train_clean.csv')

# Convert the target to 1s and 0s - models need numbers, not letters
# 1 = Bad loan (the thing we're trying to catch), 0 = Good loan
df['target'] = (df['QUALITY_OF_LOAN'] == 'B').astype(int)

# ---------------------------------------------------------
# 2. FEATURE PREP
# ---------------------------------------------------------
feature_cols = ['INVESTMENT_TOTAL', 'ACCCURRENTBALANCE', 'INSTALL_SIZE',
                 'DUE_PAYMENT', 'INF_MARITAL_STATUS', 'INF_GENDER',
                 'COMPENSATION_CHARGED', 'CLIENT_TYPE', 'REPAY_MODE']

X = df[feature_cols].copy()
y = df['target']

# Models can't read text categories like "Rural" or "M" directly -
# they need numbers. One-hot encoding turns each category into its own
# 0/1 column (e.g. CLIENT_TYPE_Urban = 1 if Urban, 0 otherwise)
X = pd.get_dummies(X, columns=['INF_MARITAL_STATUS', 'INF_GENDER',
                                 'COMPENSATION_CHARGED', 'CLIENT_TYPE',
                                 'REPAY_MODE'], drop_first=True)

# ---------------------------------------------------------
# 3. TRAIN/VALIDATION SPLIT
# ---------------------------------------------------------
# We hold back 20% of the data the model never sees during training,
# so we can test how well it performs on "new" accounts.
# stratify=y keeps the 89/11 Good/Bad ratio the same in both splits -
# otherwise a random split could accidentally put almost all the bad
# loans in one side by chance.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic regression is sensitive to feature scale (e.g. loan amounts
# in the millions vs. a 0/1 gender column) - scaling puts everything on
# a comparable range. Random Forest doesn't need this, so we only scale
# for the logistic regression model.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ---------------------------------------------------------
# 4. MODEL 1: LOGISTIC REGRESSION
# ---------------------------------------------------------
# class_weight='balanced' tells the model to pay more attention to the
# minority class (Bad loans, only 11% of data). Without this, the model
# could just predict "Good" every time and still be 89% "accurate" -
# while catching zero actual risk.
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_val_scaled)
log_probs = log_model.predict_proba(X_val_scaled)[:, 1]  # probability of "Bad"

# ---------------------------------------------------------
# 5. MODEL 2: RANDOM FOREST
# ---------------------------------------------------------
# Random Forest builds many decision trees and averages their votes.
# It can capture non-linear patterns (e.g. "risk only spikes when
# BOTH balance is low AND compensation was charged") that logistic
# regression can't easily represent.
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_probs = rf_model.predict_proba(X_val)[:, 1]

# ---------------------------------------------------------
# 6. EVALUATE
# ---------------------------------------------------------
for name, preds, probs in [('Logistic Regression', log_preds, log_probs),
                             ('Random Forest', rf_preds, rf_probs)]:
    print(f"\n--- {name} ---")
    print(confusion_matrix(y_val, preds))
    print(classification_report(y_val, preds, target_names=['Good', 'Bad']))
    print("ROC-AUC:", roc_auc_score(y_val, probs))

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------
# Shows which features the Random Forest relied on most when making
# its splits - this is what tells a lender "here's what actually
# predicts risk," not just "here's a black box score."
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))
