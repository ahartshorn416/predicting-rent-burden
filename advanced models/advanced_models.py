"""
advanced_models.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

This script fits Random Forest and Gradient Boosting classifiers to predict rent burden,
tunes hyperparameters via RandomizedSearchCV, evaluates both models using PR-AUC,
ROC-AUC, and classification metrics, and compares results against the logistic
regression baseline. All outputs are saved to the results directory.
"""
# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform

# -----------------------------
# Path Settings
# -----------------------------
path = r"C:\Users\alica\Downloads\usa_00003.csv"
output_dir = r"C:\Users\alica\OneDrive\Documents\predicting-rent-burden\results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load & Preprocess
# -----------------------------
cols_needed = ['HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT',
               'AGE', 'EDUC', 'SEX', 'RACE']

df = pd.read_csv(path, usecols=cols_needed)
print(f"Data loaded: {df.shape}")

df['HHINCOME'] = df['HHINCOME'].replace(9999999, np.nan)
df['HHINCOME'] = df['HHINCOME'].fillna(df['HHINCOME'].median())

df = df[df['RENTGRS'] > 0].copy()
print(f"Renter-only dataset: {df.shape}")

# -----------------------------
# Feature Engineering
# -----------------------------
df['rent_burdened'] = np.where(
    df['HHINCOME'] <= 0,
    1,
    np.where((df['RENTGRS'] / df['HHINCOME']) > 0.3, 1, 0)
)

df['UNSTABLE_EMPLOYMENT'] = np.where(
    (df['WKSWORK1'] < 35) | (df['EMPSTAT'].isin([2, 3])),
    1, 0
)

features = ['AGE', 'EDUC', 'SEX', 'RACE', 'WKSWORK1', 'UNSTABLE_EMPLOYMENT']
target = 'rent_burdened'

df_model = df[features + [target]].dropna()
print(f"Modeling dataset: {df_model.shape}")
print(f"Positive class rate: {df_model[target].mean():.4f}")

# -----------------------------
# Train / Test Split
# -----------------------------
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

baseline_prauc = y_train.mean()
print(f"Naive baseline PR-AUC: {baseline_prauc:.4f}")

# Load baseline LR metrics for comparison
baseline_metrics_path = os.path.join(output_dir, "baseline_metrics.csv")
if os.path.exists(baseline_metrics_path):
    baseline_df = pd.read_csv(baseline_metrics_path)
    lr_roc   = baseline_df.loc[baseline_df['Metric'] == 'ROC-AUC',  'Value'].values[0]
    lr_prauc = baseline_df.loc[baseline_df['Metric'] == 'PR-AUC',   'Value'].values[0]
    print(f"Loaded baseline LR — ROC-AUC: {lr_roc:.4f}, PR-AUC: {lr_prauc:.4f}")
else:
    lr_roc, lr_prauc = 0.7730, 0.0967
    print("Baseline metrics file not found — using reported values.")

# -----------------------------
# Subsample for hyperparameter search
# -----------------------------
# The full training set (~2.85M rows) causes timeout during RandomizedSearchCV.
# Strategy: tune on a stratified 10% subsample (~285K rows), which preserves
# the class ratio and is large enough for stable CV estimates.
# Best params are then refit on the FULL training set for final evaluation.

TUNE_FRAC = 0.10
print(f"\nSubsampling {TUNE_FRAC*100:.0f}% of training data for hyperparameter search...")

X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train,
    train_size=TUNE_FRAC,
    random_state=42,
    stratify=y_train
)
print(f"Tuning set: {X_tune.shape} | Positive rate: {y_tune.mean():.4f}")

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# Utility: evaluate a fitted model on the test set
# -----------------------------
def evaluate_model(name, model, X_test, y_test, baseline_prauc, output_dir, features):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    print(f"\n{'='*55}")
    print(f"  {name} — TEST SET RESULTS")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}  (naive baseline: {baseline_prauc:.4f})")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    slug = name.lower().replace(" ", "_")

    # Feature importances
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\nFeature Importances:\n{imp_df.to_string(index=False)}")
        imp_df.to_csv(os.path.join(output_dir, f"{slug}_importances.csv"), index=False)

        plt.figure(figsize=(8, 5))
        plt.barh(imp_df['Feature'], imp_df['Importance'])
        plt.title(f"Feature Importances — {name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{slug}_importances.png"))
        plt.close()

    # ROC Curve
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"ROC Curve — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slug}_roc_curve.png"))
    plt.close()

    # Precision-Recall Curve
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.axhline(baseline_prauc, linestyle='--', color='gray',
               label=f'Naive baseline ({baseline_prauc:.3f})')
    ax.legend()
    ax.set_title(f"Precision-Recall Curve — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slug}_pr_curve.png"))
    plt.close()

    return roc_auc, pr_auc


# ================================================================
# MODEL 1: RANDOM FOREST
# ================================================================
print("\n" + "="*60)
print("RANDOM FOREST — HYPERPARAMETER TUNING (on subsample)")
print("="*60)

rf_param_dist = {
    'n_estimators':      randint(100, 400),
    'max_depth':         [10, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf':  randint(1, 10),
    'max_features':      ['sqrt', 'log2', 0.3],
}

rf_base = RandomForestClassifier(
    class_weight='balanced_subsample',  # per-tree balancing; better for bagging
    random_state=42,
    n_jobs=-1
)

rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=rf_param_dist,
    n_iter=15,
    scoring='average_precision',  # PR-AUC; minority-class focused
    cv=cv_strat,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=False          # we refit manually on full training set below
)

rf_search.fit(X_tune, y_tune)

best_rf_params = rf_search.best_params_
print(f"\nBest RF params (from subsample search): {best_rf_params}")
print(f"Best CV PR-AUC (subsample): {rf_search.best_score_:.4f}")

# Save search results
pd.DataFrame(rf_search.cv_results_)[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
].sort_values('rank_test_score').to_csv(
    os.path.join(output_dir, "rf_search_results.csv"), index=False
)

# Refit best params on FULL training set
print("\nRefitting best RF params on full training set...")
best_rf = RandomForestClassifier(
    **best_rf_params,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train, y_train)

rf_roc, rf_prauc = evaluate_model(
    name="Random Forest",
    model=best_rf,
    X_test=X_test,
    y_test=y_test,
    baseline_prauc=baseline_prauc,
    output_dir=output_dir,
    features=features
)


# ================================================================
# MODEL 2: GRADIENT BOOSTING (HistGradientBoostingClassifier)
# ================================================================
# HistGradientBoosting is used over GradientBoostingClassifier because:
#   - Histogram-binning makes it orders of magnitude faster on large datasets
#   - Supports sample_weight natively for imbalance handling
#   - Comparable or better performance to standard GBM at this scale
# ================================================================

print("\n" + "="*60)
print("GRADIENT BOOSTING — HYPERPARAMETER TUNING (on subsample)")
print("="*60)

# Class balancing via sample_weight (equivalent to class_weight='balanced')
neg_count  = (y_tune == 0).sum()
pos_count  = (y_tune == 1).sum()
pos_weight = neg_count / pos_count
tune_weights = np.where(y_tune == 1, pos_weight, 1.0)
print(f"Positive class upweight: {pos_weight:.1f}x")

gb_param_dist = {
    'max_iter':          randint(100, 400),
    'max_depth':         randint(3, 8),
    'learning_rate':     uniform(0.01, 0.19),
    'min_samples_leaf':  randint(10, 100),
    'l2_regularization': uniform(0, 1.0),
    'max_leaf_nodes':    [15, 31, 63, None],
}

gb_base = HistGradientBoostingClassifier(random_state=42)

gb_search = RandomizedSearchCV(
    estimator=gb_base,
    param_distributions=gb_param_dist,
    n_iter=15,
    scoring='average_precision',
    cv=cv_strat,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=False
)

gb_search.fit(X_tune, y_tune, sample_weight=tune_weights)

best_gb_params = gb_search.best_params_
print(f"\nBest GB params (from subsample search): {best_gb_params}")
print(f"Best CV PR-AUC (subsample): {gb_search.best_score_:.4f}")

pd.DataFrame(gb_search.cv_results_)[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
].sort_values('rank_test_score').to_csv(
    os.path.join(output_dir, "gb_search_results.csv"), index=False
)

# Refit on FULL training set with full sample weights
print("\nRefitting best GB params on full training set...")
full_weights = np.where(
    y_train == 1,
    (y_train == 0).sum() / (y_train == 1).sum(),
    1.0
)

best_gb = HistGradientBoostingClassifier(**best_gb_params, random_state=42)
best_gb.fit(X_train, y_train, sample_weight=full_weights)

gb_roc, gb_prauc = evaluate_model(
    name="Gradient Boosting",
    model=best_gb,
    X_test=X_test,
    y_test=y_test,
    baseline_prauc=baseline_prauc,
    output_dir=output_dir,
    features=features
)


# ================================================================
# MODEL COMPARISON TABLE
# ================================================================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_df = pd.DataFrame({
    'Model':   ['Logistic Regression (Baseline)', 'Random Forest', 'Gradient Boosting'],
    'ROC-AUC': [lr_roc,   rf_roc,   gb_roc],
    'PR-AUC':  [lr_prauc, rf_prauc, gb_prauc],
    'PR-AUC lift vs naive': [
        lr_prauc / baseline_prauc,
        rf_prauc / baseline_prauc,
        gb_prauc / baseline_prauc,
    ]
})

print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

# Grouped bar chart
x     = np.arange(len(comparison_df))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, comparison_df['ROC-AUC'], width, label='ROC-AUC')
ax.bar(x + width/2, comparison_df['PR-AUC'],  width, label='PR-AUC')
ax.axhline(baseline_prauc, linestyle='--', color='gray', linewidth=0.9,
           label=f'Naive PR-AUC ({baseline_prauc:.3f})')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'], rotation=10, ha='right')
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: ROC-AUC and PR-AUC")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_comparison.png"))
plt.close()

print(f"\nAll outputs saved to: {output_dir}")
print("\n✅ Advanced models complete.")