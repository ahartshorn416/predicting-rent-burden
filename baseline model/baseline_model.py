"""
baseline_model.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

This script fits a logistic regression baseline model to predict rent burden,
evaluates it using PR-AUC, ROC-AUC, and classification metrics, performs
cross-validation, and saves all results to the output directory.
"""

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Path Settings
# -----------------------------
path = r"C:\Users\alica\Downloads\usa_00003.csv"
output_dir = r"C:\Users\alica\OneDrive\Documents\predicting-rent-burden\results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load & Preprocess
# (same pipeline as feature_engineering.py)
# -----------------------------
cols_needed = ['HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT',
               'AGE', 'EDUC', 'SEX', 'RACE']

df = pd.read_csv(path, usecols=cols_needed)
print(f"Data loaded: {df.shape}")

# Sentinel value + median impute
df['HHINCOME'] = df['HHINCOME'].replace(9999999, np.nan)
df['HHINCOME'] = df['HHINCOME'].fillna(df['HHINCOME'].median())

# Restrict to renters
df = df[df['RENTGRS'] > 0].copy()
print(f"Renter-only dataset: {df.shape}")

# -----------------------------
# Feature Engineering
# -----------------------------
# Target variable
df['rent_burdened'] = np.where(
    df['HHINCOME'] <= 0,
    1,
    np.where((df['RENTGRS'] / df['HHINCOME']) > 0.3, 1, 0)
)

# Log transforms
df['log_income'] = np.log1p(df['HHINCOME'].clip(lower=0))
df['log_rent'] = np.log1p(df['RENTGRS'].clip(lower=0))

# Rent-to-income ratio
df['rent_income_ratio'] = df['RENTGRS'] / (df['HHINCOME'] + 1)

# Employment instability
df['UNSTABLE_EMPLOYMENT'] = np.where(
    (df['WKSWORK1'] < 35) | (df['EMPSTAT'].isin([2, 3])),
    1, 0
)

features = [
    'AGE',
    'EDUC',
    'SEX',
    'RACE',
    'WKSWORK1',
    'UNSTABLE_EMPLOYMENT'
]
target = 'rent_burdened'

df_model = df[features + [target]].dropna()
print(f"Modeling dataset: {df_model.shape}")
print(f"Positive class rate: {df_model[target].mean():.4f}")

# -----------------------------
# Train / Test Split
# (use saved CSVs if available, otherwise re-split)
# -----------------------------
from sklearn.model_selection import train_test_split

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Positive Class Baseline (naive)
# -----------------------------
baseline_prauc = y_train.mean()
print(f"\nNaive positive-class baseline PR-AUC: {baseline_prauc:.4f}")

# -----------------------------
# Logistic Regression
# Chosen as baseline because:
#   - Interpretable coefficients for each predictor
#   - Outputs calibrated probabilities (needed for PR-AUC)
#   - Standard baseline for binary classification
#   - class_weight='balanced' handles class imbalance without resampling
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # addresses class imbalance
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
print("\n--- CLASSIFICATION REPORT ---")
report = classification_report(y_test, y_pred)
print(report)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"ROC-AUC:  {roc_auc:.4f}")
print(f"PR-AUC:   {pr_auc:.4f}  (baseline: {baseline_prauc:.4f})")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': ['ROC-AUC', 'PR-AUC', 'Naive Baseline PR-AUC'],
    'Value': [roc_auc, pr_auc, baseline_prauc]
})
metrics_df.to_csv(os.path.join(output_dir, "baseline_metrics.csv"), index=False)
print("\nMetrics saved.")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Coefficients
# -----------------------------
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print("\n--- COEFFICIENTS ---")
print(coef_df.to_string(index=False))
coef_df.to_csv(os.path.join(output_dir, "baseline_coefficients.csv"), index=False)

# Plot coefficients
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
ax.set_title("ROC Curve — Logistic Regression Baseline")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "baseline_roc_curve.png"))
plt.close()

# -----------------------------
# Precision-Recall Curve
# -----------------------------
fig, ax = plt.subplots()
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.axhline(baseline_prauc, linestyle='--', color='gray', label=f'Naive baseline ({baseline_prauc:.3f})')
ax.legend()
ax.set_title("Precision-Recall Curve — Logistic Regression Baseline")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "baseline_pr_curve.png"))
plt.close()

# -----------------------------
# Cross-Validation (StratifiedKFold, k=5)
# Stratified to preserve class imbalance ratio across folds
# -----------------------------
print("\n--- CROSS-VALIDATION (k=5) ---")

cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    cv_pipeline,
    X_train,        # ← line 223: change X_train_scaled to X_train
    y_train,
    cv=cv,
    scoring=['roc_auc', 'average_precision'],
    return_train_score=True
)

cv_df = pd.DataFrame({
    'Fold': range(1, 6),
    'Train ROC-AUC': cv_results['train_roc_auc'],
    'Val ROC-AUC': cv_results['test_roc_auc'],
    'Train PR-AUC': cv_results['train_average_precision'],
    'Val PR-AUC': cv_results['test_average_precision'],
})
print(cv_df.to_string(index=False))
print(f"\nMean Val ROC-AUC: {cv_df['Val ROC-AUC'].mean():.4f} (+/- {cv_df['Val ROC-AUC'].std():.4f})")
print(f"Mean Val PR-AUC:  {cv_df['Val PR-AUC'].mean():.4f} (+/- {cv_df['Val PR-AUC'].std():.4f})")

cv_df.to_csv(os.path.join(output_dir, "baseline_cv_results.csv"), index=False)

print("\n✅ Baseline model complete — all outputs saved.")