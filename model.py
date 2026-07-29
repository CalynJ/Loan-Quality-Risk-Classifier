"""
Loan Quality Prediction Model
==============================
Predicts whether a loan is "Good" (G) or "Bad" (B) based on account
and borrower features, using logistic regression and random forest,
with class imbalance handling.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, ConfusionMatrixDisplay)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv('/home/claude/project/train_clean.csv')

# Target: 1 = Bad loan, 0 = Good loan
df['target'] = (df['QUALITY_OF_LOAN'] == 'B').astype(int)

print("Class balance:")
print(df['target'].value_counts(normalize=True))

# ---------------------------------------------------------
# 2. FEATURE PREP
# ---------------------------------------------------------
feature_cols = ['INVESTMENT_TOTAL', 'ACCCURRENTBALANCE', 'INSTALL_SIZE',
                 'DUE_PAYMENT', 'INF_MARITAL_STATUS', 'INF_GENDER',
                 'COMPENSATION_CHARGED', 'CLIENT_TYPE', 'REPAY_MODE']

X = df[feature_cols].copy()
y = df['target']

# One-hot encode categoricals
X = pd.get_dummies(X, columns=['INF_MARITAL_STATUS', 'INF_GENDER',
                                 'COMPENSATION_CHARGED', 'CLIENT_TYPE',
                                 'REPAY_MODE'], drop_first=True)

print(f"\nFinal feature set ({X.shape[1]} features):")
print(list(X.columns))

# ---------------------------------------------------------
# 3. TRAIN/VALIDATION SPLIT
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ---------------------------------------------------------
# 4. MODEL 1: LOGISTIC REGRESSION (class_weight='balanced' handles the
#    11% bad-loan imbalance by upweighting the minority class)
# ---------------------------------------------------------
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_val_scaled)
log_probs = log_model.predict_proba(X_val_scaled)[:, 1]

# ---------------------------------------------------------
# 5. MODEL 2: RANDOM FOREST
# ---------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_probs = rf_model.predict_proba(X_val)[:, 1]

# ---------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------
results = {}
for name, preds, probs in [('Logistic Regression', log_preds, log_probs),
                             ('Random Forest', rf_preds, rf_probs)]:
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    cm = confusion_matrix(y_val, preds)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y_val, preds, target_names=['Good', 'Bad']))
    auc = roc_auc_score(y_val, probs)
    print(f"ROC-AUC: {auc:.4f}")
    results[name] = {'cm': cm, 'auc': auc, 'probs': probs}

# ---------------------------------------------------------
# 7. VISUALIZATIONS
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrices
for ax, (name, res) in zip(axes[:2], results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=res['cm'],
                                    display_labels=['Good', 'Bad'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"{name}\nROC-AUC: {res['auc']:.3f}")

# ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_val, res['probs'])
    axes[2].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curves')
axes[2].legend()

plt.tight_layout()
plt.savefig('/home/claude/project/model_evaluation.png', dpi=150)
print("\nSaved model_evaluation.png")

# ---------------------------------------------------------
# 8. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8, 6))
importances.head(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.title('Top 10 Risk Factors (Random Forest)')
plt.tight_layout()
plt.savefig('/home/claude/project/feature_importance.png', dpi=150)
print("Saved feature_importance.png")

print("\nTop 10 features driving bad-loan predictions:")
print(importances.head(10))
