# Loan Quality Risk Classifier

Predicts whether a microfinance loan account will be classified as **Good (G)** or **Bad (B)**
based on account balance, investment size, borrower demographics, and repayment behavior.

## Business Problem

Lenders need to identify high-risk accounts *before* they default, not after. This project
builds a binary classifier on ~37K real loan accounts to flag likely "Bad" loans, using
features available at/near account setup (loan size, balance, client type, repayment mode).

The dataset has a real-world imbalance: **only 11.1% of loans are labeled "Bad."** This
mirrors actual credit risk data, where defaults are the minority outcome — and it's the
central modeling challenge of the project.

## Data

- **37,298 training accounts** / 4,310 test accounts (after cleaning)
- Features: `INVESTMENT_TOTAL`, `ACCCURRENTBALANCE`, `INSTALL_SIZE`, `DUE_PAYMENT`,
  `INF_MARITAL_STATUS`, `INF_GENDER`, `COMPENSATION_CHARGED`, `CLIENT_TYPE` (Rural/Semi-urban/Urban),
  `REPAY_MODE`
- Target: `QUALITY_OF_LOAN` (G/B)

### Cleaning steps
- Removed 3 rows with invalid `"0"` placeholder values in `CLIENT_TYPE`
- Dropped ~100 rows with missing categorical fields (negligible loss relative to 37K rows)
- Filled missing `INSTALL_SIZE` with 0, consistent with the column's dominant value

## Methodology

1. **One-hot encoded** categorical features (marital status, gender, client type, repayment mode)
2. **Stratified 80/20 train/validation split** to preserve the 11% bad-loan rate in both sets
3. Trained two models:
   - **Logistic Regression** (baseline, interpretable, industry-standard for credit scoring)
   - **Random Forest** (captures non-linear interactions between features)
4. Both models used **`class_weight='balanced'`** to counteract the 89/11 class imbalance —
   without this, a naive model could hit 89% accuracy by simply predicting "Good" every time
   and never catch a single bad loan.

## Why ROC-AUC and Recall, Not Accuracy

In a risk model like this, the cost of mistakes is **asymmetric**:
- Missing a Bad loan (false negative) costs the lender real money
- Flagging a Good loan as risky (false positive) costs a customer relationship, but far less

Because of this, **accuracy is a misleading metric** here (predicting "all Good" would score 89%
accuracy while catching zero risk). Instead, this project evaluates on:
- **Recall on the Bad class** — how many actual bad loans the model catches
- **ROC-AUC** — how well the model separates good from bad accounts across all thresholds

## Results

| Model | ROC-AUC | Recall (Bad) | Precision (Bad) |
|---|---|---|---|
| Logistic Regression | 0.674 | 0.67 | 0.18 |
| **Random Forest** | **0.757** | 0.67 | 0.21 |

**Random Forest is the stronger model**, with a meaningfully higher ROC-AUC (0.757 vs. 0.674),
meaning it separates good and bad loans more reliably across decision thresholds. Both models
catch about 67% of bad loans at this threshold — a deliberate choice favoring risk-catching over
precision, appropriate for a risk-averse lending context.

Precision on the "Bad" class is low (18-21%) for both models. This is an expected tradeoff
in imbalanced classification: to catch two-thirds of true bad loans, the model also flags a
number of loans that are actually fine. In a production setting, this is a **business decision**,
not just a modeling one — a lender may accept more false alarms (which trigger manual review)
in exchange for catching more real risk.

## Top Risk Drivers (Random Forest Feature Importance)

1. **Compensation charged (Y/N)** — whether a penalty/compensation was ever charged is the single
   strongest predictor of loan quality — an early warning sign already present in the data.
2. **Account current balance** — outstanding balance size
3. **Investment total** — original loan/investment size
4. **Urban client type** — urban accounts behave differently than rural/semi-urban ones
5. **Installment size**

## What I'd Do Next

- Engineer a ratio feature (balance / investment total) to capture repayment progress directly
- Try `class_weight` tuning or SMOTE oversampling to compare against `balanced` weighting
- Test a gradient boosting model (XGBoost/LightGBM) for a likely AUC improvement over Random Forest
- If deployed, calibrate the decision threshold against the lender's actual cost of false
  positives vs. false negatives, rather than using the default 0.5 cutoff

## Tools

Python, pandas, scikit-learn (LogisticRegression, RandomForestClassifier), matplotlib

---
*Note: dataset is anonymized microfinance loan account data. Account numbers and borrower
identities are not personally identifiable.*
