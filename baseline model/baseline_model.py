"""
baseline_model.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

This script runs the logistic regression baseline model. I started with logistic
regression because it gives me a clear performance floor to compare against when
I move on to random forest and gradient boosting. It also outputs probabilities
rather than just class labels, which I needed to calculate PR-AUC. And the
coefficients are interpretable, which is useful for checking whether the model
is picking up on what I'd expect it to pick up on.

The model uses only demographic and employment features. income and rent are
excluded because including them gave ROC-AUC = 1.0 (leakage). See
preprocessing.py for details on that.

Roles:
- Load & Preprocess: AH
- Train / Test Split: AH
- Logistic Regression Model: AH
- Evaluation Metrics: AH
- Cross-Validation: AH
- Testing & Validation: AH
"""

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# preprocessing.py is at the repo root, one level up from this subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import (
    load_and_clean,
    build_features,
    get_feature_list,
    get_target,
)

# -----------------------------
# Path Settings
# -----------------------------
path       = r"C:\Users\alica\Downloads\usa_00003.csv"
output_dir = r"C:\Users\alica\OneDrive\Documents\predicting-rent-burden\results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load & Preprocess
# -----------------------------
# Same pipeline as feature_engineering.py. Calling the shared functions
# so I know everything is consistent across scripts.
cols_needed = [
    'HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT',
    'AGE', 'EDUC', 'SEX', 'RACE'
]
df = load_and_clean(path, cols_needed=cols_needed)
df = build_features(df)

features = get_feature_list()
target   = get_target()

df_model = df[features + [target]].dropna()
print(f"Modeling dataset: {df_model.shape}")
print(f"Positive class rate: {df_model[target].mean():.4f}")

# -----------------------------
# Train / Test Split
# -----------------------------
X = df_model[features]
y = df_model[target]

# Same split as feature_engineering.py. random_state=42 and stratify=y
# so the train/test sets are identical across all modeling scripts.
# stratify=y makes sure both sets keep the 3.65% positive rate, which
# matters a lot when the positive class is this small.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler on training data only, then apply to test.
# If I fit on the full dataset first the test set's distribution
# influences the scaling, which is leakage.
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------
# Naive Baseline PR-AUC
# -----------------------------
# The floor I need to beat. A model that always predicts the majority
# class would get PR-AUC equal to the positive class rate (3.65%).
# Anything I build has to meaningfully exceed this to be worth using.
baseline_prauc = y_train.mean()
print(f"\nNaive positive-class baseline PR-AUC: {baseline_prauc:.4f}")

# -----------------------------
# Logistic Regression Model
# -----------------------------
# I used class_weight='balanced' because without it the model just predicts
# all-zero with a 3.65% positive rate, that gets 96.4% accuracy and
# misses every single rent-burdened household. Balanced weighting fixes
# this by upweighting the minority class during training.
#
# The tradeoff is that precision drops. The model flags a lot of
# false positives. But for this problem, missing an at-risk household
# is a worse error than flagging one that turns out to be fine, so
# I'm okay with that at the baseline stage.
#
# solver='lbfgs' is the default for this size of dataset. max_iter=1000
# is higher than the default because the model was throwing convergence
# warnings at 100 iterations on the 2.85M training rows.
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'
)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# Evaluation Metrics
# -----------------------------
# I'm treating PR-AUC as the primary metric rather than ROC-AUC or accuracy.
# With a 3.65% positive rate, accuracy is misleading (predicting all zeros
# scores 96.4%) and ROC-AUC can look good even when the model is struggling
# with the minority class. PR-AUC measures exactly what I care about which is
# how well the model handles the rent-burdened households.
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)

print(f"ROC-AUC:  {roc_auc:.4f}")
print(f"PR-AUC:   {pr_auc:.4f}  (naive baseline: {baseline_prauc:.4f})")

metrics_df = pd.DataFrame({
    'Metric': ['ROC-AUC', 'PR-AUC', 'Naive Baseline PR-AUC'],
    'Value':  [roc_auc, pr_auc, baseline_prauc]
})
metrics_df.to_csv(os.path.join(output_dir, "baseline_metrics.csv"), index=False)
print("\nMetrics saved.")

# -----------------------------
# Confusion Matrix
# -----------------------------
# With class_weight='balanced' I expected high recall and low precision,
# and that's what came back. The model catches most rent-burdened
# households but flags a lot of non-burdened ones too. That's the
# direct result of upweighting the minority class.
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Coefficients
# -----------------------------
# One of the main reasons I started with logistic regression is that
# the coefficients tell me whether the model is picking up on what I'd
# expect. WKSWORK1 came back as the strongest predictor (most negative),
# which is exactly what the hypothesis predicts, more weeks worked
# means lower rent burden risk. The positive coefficient on EDUC was
# unexpected and worth looking into in the next phase.
coef_df = pd.DataFrame({
    'Feature':     features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print("\n--- COEFFICIENTS ---")
print(coef_df.to_string(index=False))
coef_df.to_csv(os.path.join(output_dir, "baseline_coefficients.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "baseline_coefficients.png"))
plt.close()

# -----------------------------
# ROC Curve
# -----------------------------
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title("ROC Curve -- Logistic Regression Baseline")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "baseline_roc_curve.png"))
plt.close()

# -----------------------------
# Precision-Recall Curve
# -----------------------------
# I care more about this plot than the ROC curve for this problem.
# The dashed line shows the naive baseline anything above it means
# the model is adding real value over just predicting the majority class.
fig, ax = plt.subplots()
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.axhline(
    baseline_prauc, linestyle='--', color='gray',
    label=f'Naive baseline ({baseline_prauc:.3f})'
)
ax.legend()
ax.set_title("Precision-Recall Curve -- Logistic Regression Baseline")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "baseline_pr_curve.png"))
plt.close()

# -----------------------------
# Cross-Validation (StratifiedKFold, k=5)
# -----------------------------
# I ran cross-validation to check whether the model was overfitting.
# I used StratifiedKFold because with this much class imbalance, regular
# KFold could give me folds with almost no positive examples, which would
# make the fold-level metrics unreliable.
#
# I wrapped the model in a Pipeline with its own scaler so that each fold's
# scaler only sees that fold's training data. If I passed X_train_scaled in
# directly, the scaler would have already seen all folds during fitting,
# which is a subtle form of leakage.
print("\n--- CROSS-VALIDATION (k=5) ---")

cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Passing raw X_train here (not X_train_scaled). The Pipeline handles
# scaling internally per fold so there's no leakage.
cv_results = cross_validate(
    cv_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=['roc_auc', 'average_precision'],
    return_train_score=True
)

cv_df = pd.DataFrame({
    'Fold':          range(1, 6),
    'Train ROC-AUC': cv_results['train_roc_auc'],
    'Val ROC-AUC':   cv_results['test_roc_auc'],
    'Train PR-AUC':  cv_results['train_average_precision'],
    'Val PR-AUC':    cv_results['test_average_precision'],
})
print(cv_df.to_string(index=False))
print(f"\nMean Val ROC-AUC: "
      f"{cv_df['Val ROC-AUC'].mean():.4f} "
      f"(+/- {cv_df['Val ROC-AUC'].std():.4f})")
print(f"Mean Val PR-AUC:  "
      f"{cv_df['Val PR-AUC'].mean():.4f} "
      f"(+/- {cv_df['Val PR-AUC'].std():.4f})")

cv_df.to_csv(os.path.join(output_dir, "baseline_cv_results.csv"), index=False)

print("\nBaseline model complete -- all outputs saved.")

# -----------------------------
# Testing & Validation
# -----------------------------

"""
SELF TEST: AH on Local Machine
------------------------------
- Script runs without errors in PyCharm
- Row count drops from 16.1M to 3.56M after renter filter, as expected
- Positive class rate: 3.65% matches feature_engineering.py exactly
- class_weight='balanced' working as expected: recall for class 1 = 0.92,
  precision is low (0.07) which is the expected tradeoff
- ROC-AUC: 0.773, PR-AUC: 0.097 (~2.6x above naive baseline of 0.037)
- Confusion matrix shows high recall / low precision as expected
- CV pipeline uses raw X_train, not X_train_scaled scaler fits within
  each fold so there's no leakage into the CV scores
- CV scores stable across all 5 folds (SD = 0.002) no sign of overfitting
- WKSWORK1 coefficient is the most negative, consistent with hypothesis
- EDUC coefficient is positive, which was unexpected worth investigating
- All output files saved: metrics CSV, coefficients CSV, 3 plots

USER TEST: Secondary Device
---------------------------
- Script is portable -- only path variables need updating
- Raw IPUMS data not in repo due to file size
- Download from: https://usa.ipums.org/usa/

REQUIRED USER STEPS:
1. Run preprocessing.py is at the repo root (one level above this folder)
2. Download 2024 ACS 5-Year dataset from IPUMS: https://usa.ipums.org/usa/
   Variables: HHINCOME, RENTGRS, WKSWORK1, EMPSTAT, AGE, EDUC, SEX, RACE
3. Update `path` to your local CSV file location
4. Update `output_dir` to your results directory
5. Install: pandas, numpy, matplotlib, scikit-learn

USER TEST STATUS: PASSED (with dataset dependency noted)
"""