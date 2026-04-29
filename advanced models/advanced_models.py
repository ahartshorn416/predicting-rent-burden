"""
advanced_models.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

Fits Random Forest and Gradient Boosting classifiers. Hyperparameter search runs on
a 5% stratified subsample; best params are refit on the full training set.
RF is capped at 150 trees to keep runtime manageable on 2.85M rows.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings, time
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint, uniform

# -----------------------------
# Paths
# -----------------------------
path       = r"C:\Users\alica\Downloads\usa_00003.csv"
output_dir = r"C:\Users\alica\OneDrive\Documents\predicting-rent-burden\results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load & Preprocess
# -----------------------------
print("Loading data...", flush=True)
cols_needed = ['HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT', 'AGE', 'EDUC', 'SEX', 'RACE']
df = pd.read_csv(path, usecols=cols_needed)
print(f"  Loaded: {df.shape}", flush=True)

df['HHINCOME'] = df['HHINCOME'].replace(9999999, np.nan)
df['HHINCOME'] = df['HHINCOME'].fillna(df['HHINCOME'].median())
df = df[df['RENTGRS'] > 0].copy()
print(f"  Renters only: {df.shape}", flush=True)

# -----------------------------
# Feature Engineering
# -----------------------------
df['rent_burdened'] = np.where(
    df['HHINCOME'] <= 0, 1,
    np.where((df['RENTGRS'] / df['HHINCOME']) > 0.3, 1, 0)
)
df['UNSTABLE_EMPLOYMENT'] = np.where(
    (df['WKSWORK1'] < 35) | (df['EMPSTAT'].isin([2, 3])), 1, 0
)

features = ['AGE', 'EDUC', 'SEX', 'RACE', 'WKSWORK1', 'UNSTABLE_EMPLOYMENT']
target   = 'rent_burdened'

df_model = df[features + [target]].dropna()
print(f"  Modeling dataset: {df_model.shape}", flush=True)
print(f"  Positive class rate: {df_model[target].mean():.4f}", flush=True)

# -----------------------------
# Train / Test Split
# -----------------------------
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape} | Test: {X_test.shape}", flush=True)

baseline_prauc = y_train.mean()
print(f"  Naive baseline PR-AUC: {baseline_prauc:.4f}", flush=True)

# Load LR baseline for comparison table
baseline_path = os.path.join(output_dir, "baseline_metrics.csv")
if os.path.exists(baseline_path):
    bdf     = pd.read_csv(baseline_path)
    lr_roc  = bdf.loc[bdf['Metric'] == 'ROC-AUC', 'Value'].values[0]
    lr_pr   = bdf.loc[bdf['Metric'] == 'PR-AUC',  'Value'].values[0]
    print(f"  Baseline LR — ROC-AUC: {lr_roc:.4f}, PR-AUC: {lr_pr:.4f}", flush=True)
else:
    lr_roc, lr_pr = 0.7730, 0.0967
    print("  Baseline metrics file not found — using reported values.", flush=True)

# -----------------------------
# 5% tuning subsample
# -----------------------------
TUNE_FRAC = 0.05
print(f"\nCreating {TUNE_FRAC*100:.0f}% tuning subsample...", flush=True)
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train, train_size=TUNE_FRAC, random_state=42, stratify=y_train
)
print(f"  Tuning set: {X_tune.shape} | Positive rate: {y_tune.mean():.4f}", flush=True)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# Evaluate helper
# -----------------------------
def evaluate(name, model, X_te, y_te, baseline_prauc, output_dir, features):
    print(f"\nEvaluating {name} on test set...", flush=True)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    roc = roc_auc_score(y_te, y_prob)
    pr  = average_precision_score(y_te, y_prob)

    print(f"\n{'='*50}\n  {name} — TEST RESULTS\n{'='*50}", flush=True)
    print(classification_report(y_te, y_pred), flush=True)
    print(f"ROC-AUC : {roc:.4f}", flush=True)
    print(f"PR-AUC  : {pr:.4f}  (naive baseline: {baseline_prauc:.4f})", flush=True)
    print(f"Confusion Matrix:\n{confusion_matrix(y_te, y_pred)}", flush=True)

    slug = name.lower().replace(" ", "_")

    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}
                           ).sort_values('Importance', ascending=False)
        print(f"\nFeature Importances:\n{imp.to_string(index=False)}", flush=True)
        imp.to_csv(os.path.join(output_dir, f"{slug}_importances.csv"), index=False)
        plt.figure(figsize=(8, 5))
        plt.barh(imp['Feature'], imp['Importance'])
        plt.title(f"Feature Importances — {name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{slug}_importances.png"))
        plt.close()

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_te, y_prob, ax=ax)
    ax.set_title(f"ROC Curve — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slug}_roc_curve.png"))
    plt.close()

    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_te, y_prob, ax=ax)
    ax.axhline(baseline_prauc, linestyle='--', color='gray',
               label=f'Naive baseline ({baseline_prauc:.3f})')
    ax.legend()
    ax.set_title(f"Precision-Recall Curve — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slug}_pr_curve.png"))
    plt.close()

    return roc, pr


# ================================================================
# MODEL 1: RANDOM FOREST
# ================================================================
print("\n" + "="*60, flush=True)
print("RANDOM FOREST — searching on subsample (n_iter=10)...", flush=True)
print("="*60, flush=True)

rf_params = {
    'n_estimators':      [100, 150],          # capped — 400 trees on 2.85M rows is too slow
    'max_depth':         [10, 20, None],
    'min_samples_split': randint(2, 15),
    'min_samples_leaf':  randint(1, 8),
    'max_features':      ['sqrt', 'log2'],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1),
    param_distributions=rf_params,
    n_iter=10,
    scoring='average_precision',
    cv=cv5,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=False
)

t0 = time.time()
rf_search.fit(X_tune, y_tune)
print(f"  Search done in {(time.time()-t0)/60:.1f} min", flush=True)
print(f"  Best params: {rf_search.best_params_}", flush=True)
print(f"  Best CV PR-AUC: {rf_search.best_score_:.4f}", flush=True)

pd.DataFrame(rf_search.cv_results_)[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
].sort_values('rank_test_score').to_csv(
    os.path.join(output_dir, "rf_search_results.csv"), index=False
)

# Cap n_estimators at 150 for the full refit to keep runtime under ~10 min
best_rf_params = rf_search.best_params_.copy()
best_rf_params['n_estimators'] = min(best_rf_params['n_estimators'], 150)

print(f"\nRefitting RF on full training set (n_estimators={best_rf_params['n_estimators']})...", flush=True)
t0 = time.time()
best_rf = RandomForestClassifier(
    **best_rf_params,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1,
    verbose=1       # prints one line per tree batch
)
best_rf.fit(X_train, y_train)
print(f"  RF refit done in {(time.time()-t0)/60:.1f} min", flush=True)

rf_roc, rf_pr = evaluate("Random Forest", best_rf, X_test, y_test,
                          baseline_prauc, output_dir, features)


# ================================================================
# MODEL 2: GRADIENT BOOSTING (HistGradientBoosting)
# ================================================================
print("\n" + "="*60, flush=True)
print("GRADIENT BOOSTING — searching on subsample (n_iter=10)...", flush=True)
print("="*60, flush=True)

pos_w        = (y_tune == 0).sum() / (y_tune == 1).sum()
tune_weights = np.where(y_tune == 1, pos_w, 1.0)
print(f"  Positive class upweight: {pos_w:.1f}x", flush=True)

gb_params = {
    'max_iter':          [100, 200, 300],
    'max_depth':         randint(3, 7),
    'learning_rate':     uniform(0.02, 0.18),
    'min_samples_leaf':  randint(10, 80),
    'l2_regularization': uniform(0, 0.8),
    'max_leaf_nodes':    [31, 63, None],
}

gb_search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    param_distributions=gb_params,
    n_iter=10,
    scoring='average_precision',
    cv=cv5,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=False
)

t0 = time.time()
gb_search.fit(X_tune, y_tune, sample_weight=tune_weights)
print(f"  Search done in {(time.time()-t0)/60:.1f} min", flush=True)
print(f"  Best params: {gb_search.best_params_}", flush=True)
print(f"  Best CV PR-AUC: {gb_search.best_score_:.4f}", flush=True)

pd.DataFrame(gb_search.cv_results_)[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
].sort_values('rank_test_score').to_csv(
    os.path.join(output_dir, "gb_search_results.csv"), index=False
)

print(f"\nRefitting GB on full training set...", flush=True)
full_w  = np.where(y_train == 1, (y_train == 0).sum() / (y_train == 1).sum(), 1.0)
t0 = time.time()
best_gb = HistGradientBoostingClassifier(**gb_search.best_params_, random_state=42)
best_gb.fit(X_train, y_train, sample_weight=full_w)
print(f"  GB refit done in {(time.time()-t0)/60:.1f} min", flush=True)

gb_roc, gb_pr = evaluate("Gradient Boosting", best_gb, X_test, y_test,
                          baseline_prauc, output_dir, features)


# ================================================================
# COMPARISON TABLE
# ================================================================
print("\n" + "="*60, flush=True)
print("MODEL COMPARISON", flush=True)
print("="*60, flush=True)

comp = pd.DataFrame({
    'Model':               ['Logistic Regression (Baseline)', 'Random Forest', 'Gradient Boosting'],
    'ROC-AUC':             [lr_roc,  rf_roc,  gb_roc],
    'PR-AUC':              [lr_pr,   rf_pr,   gb_pr],
    'PR-AUC lift vs naive':[lr_pr/baseline_prauc, rf_pr/baseline_prauc, gb_pr/baseline_prauc],
})
print(comp.to_string(index=False), flush=True)
comp.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

x, w = np.arange(3), 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, comp['ROC-AUC'], w, label='ROC-AUC')
ax.bar(x + w/2, comp['PR-AUC'],  w, label='PR-AUC')
ax.axhline(baseline_prauc, linestyle='--', color='gray',
           label=f'Naive PR-AUC ({baseline_prauc:.3f})')
ax.set_xticks(x)
ax.set_xticklabels(comp['Model'], rotation=10, ha='right')
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: ROC-AUC and PR-AUC")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_comparison.png"))
plt.close()

print(f"\nAll outputs saved to: {output_dir}", flush=True)
print("\n✅ Advanced models complete.", flush=True)