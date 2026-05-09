"""
advanced_models.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

This script runs Random Forest and Gradient Boosting as followups to the
logistic regression baseline. Both models did meaningfully better. PR-AUC
went from 0.097 to around 0.148, which confirmed there's real non-linear
signal in the employment and demographic features that logistic regression
couldn't capture.

A few things worth flagging about how this is set up:

- RandomizedSearchCV on the full 2.85M training set timed out, so I ran
  the hyperparameter search on a stratified 5% subsample and then refit
  the best parameters on the full training set. The subsample CV estimates
  were close enough to the final test results that I'm confident it found
  reasonable parameters.

- For Random Forest I used class_weight='balanced_subsample' instead of
  'balanced'. With bagging, applying one global weight to every tree doesn't
  make much sense since each tree sees a different bootstrap sample.
  balanced_subsample rebalances within each tree, which is a better fit.

- For Gradient Boosting I had to write a custom BalancedHGB wrapper class.
  When I passed sample_weight through fit_params to RandomizedSearchCV,
  sklearn was applying those weights during scoring too, not just training.
  That inflated CV PR-AUC from ~0.15 to ~0.78, which was obviously wrong.
  Moving the weight computation inside fit() fixed it.

- Fairness analysis runs on the best model (Gradient Boosting) using
  PPR, TPR, and FPR per RACE and SEX subgroup. White and Male are the
  reference groups.

Roles:
- Load & Preprocess: AH
- Tuning Subsample: AH
- Random Forest: AH
- Gradient Boosting / BalancedHGB: AH
- Model Comparison: AH
- Fairness Analysis: AH
- Testing & Validation: AH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import warnings
import time

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
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import randint, uniform

# preprocessing.py is at the repo root, one level up from this subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import (
    load_and_clean,
    build_features,
    get_feature_list,
    get_target,
)

# -----------------------------
# Paths
# -----------------------------
path       = r"C:\Users\alica\Downloads\usa_00003.csv"
output_dir = r"C:\Users\alica\OneDrive\Documents\predicting-rent-burden\results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load & Preprocess
# -----------------------------
# Same pipeline as baseline_model.py using the shared functions so
# I know the data going into these models is identical to what the
# baseline trained on.
print("Loading data...", flush=True)

cols_needed = [
    'HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT',
    'AGE', 'EDUC', 'SEX', 'RACE'
]
df = load_and_clean(path, cols_needed=cols_needed)
df = build_features(df)
print(f"  Renters only: {df.shape}", flush=True)

features = get_feature_list()
target   = get_target()

df_model = df[features + [target]].dropna()
print(f"  Modeling dataset: {df_model.shape}", flush=True)
print(f"  Positive class rate: {df_model[target].mean():.4f}", flush=True)

# -----------------------------
# Train / Test Split
# -----------------------------
X = df_model[features]
y = df_model[target]

# Same random_state=42 and stratify=y as baseline_model.py so all three
# models are evaluated on exactly the same test set. That's important for
# the comparison table to be meaningful.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape} | Test: {X_test.shape}", flush=True)

baseline_prauc = y_train.mean()
print(f"  Naive baseline PR-AUC: {baseline_prauc:.4f}", flush=True)

# Load the LR baseline metrics so I can include them in the comparison table.
# If baseline_model.py hasn't been run yet the CSV won't exist, so I fall
# back to the values from my baseline report.
baseline_path = os.path.join(output_dir, "baseline_metrics.csv")
if os.path.exists(baseline_path):
    bdf    = pd.read_csv(baseline_path)
    lr_roc = bdf.loc[bdf['Metric'] == 'ROC-AUC', 'Value'].values[0]
    lr_pr  = bdf.loc[bdf['Metric'] == 'PR-AUC',  'Value'].values[0]
    print(f"  Baseline LR -- ROC-AUC: {lr_roc:.4f}, PR-AUC: {lr_pr:.4f}",
          flush=True)
else:
    lr_roc, lr_pr = 0.7730, 0.0967
    print("  baseline_metrics.csv not found -- using reported values.",
          flush=True)

# -----------------------------
# 5% Tuning Subsample
# -----------------------------
# When I first ran RandomizedSearchCV on the full training set it timed out
# after about 3 hours. I switched to a stratified 5% subsample (~142K rows)
# which runs the search in about 5 minutes and still gives stable CV estimates
# with only 6 features. Once I have the best parameters I refit on the full
# training set, so the final models still train on everything.
#
# I checked that the subsample was representative enough by comparing the
# subsample CV PR-AUC for RF (0.135) to the final test set PR-AUC (0.146)
# they're close enough that I'm confident the subsample found good parameters.
TUNE_FRAC = 0.05
print(f"\nCreating {TUNE_FRAC*100:.0f}% tuning subsample...", flush=True)
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train,
    train_size=TUNE_FRAC,
    random_state=42,
    stratify=y_train
)
print(f"  Tuning set: {X_tune.shape} | "
      f"Positive rate: {y_tune.mean():.4f}", flush=True)

# Stratified folds for the hyperparameter search. Same reason as everywhere
# else, I need each fold to keep the 3.65% positive rate.
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# -----------------------------
# Evaluate Helper Function
# -----------------------------
def evaluate(name, model, X_te, y_te, baseline_prauc, output_dir, features):
    """
    Runs evaluation on the held-out test set and saves all the outputs.

    I pulled this into a function because I'm running it for both RF and GB
    and didn't want to duplicate 40 lines twice. It handles the classification
    report, confusion matrix, feature importances (if available), and both
    the ROC and PR curve plots.

    Args:
        name (str):             Model name for plot titles and file names.
        model:                  Fitted sklearn-compatible classifier.
        X_te (pd.DataFrame):    Test features.
        y_te (pd.Series):       Test labels.
        baseline_prauc (float): Naive baseline for the PR curve reference line.
        output_dir (str):       Where to save output files.
        features (list):        Feature names for labeling importances.

    Returns:
        tuple: (roc_auc, pr_auc) for the comparison table.
    """
    print(f"\nEvaluating {name} on test set...", flush=True)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    roc = roc_auc_score(y_te, y_prob)
    pr  = average_precision_score(y_te, y_prob)

    print(f"\n{'='*50}\n  {name} -- TEST RESULTS\n{'='*50}", flush=True)
    print(classification_report(y_te, y_pred), flush=True)
    print(f"ROC-AUC : {roc:.4f}", flush=True)
    print(f"PR-AUC  : {pr:.4f}  (naive baseline: {baseline_prauc:.4f})",
          flush=True)
    print(f"Confusion Matrix:\n{confusion_matrix(y_te, y_pred)}", flush=True)

    slug = name.lower().replace(" ", "_")

    # RF exposes feature importances directly; BalancedHGB doesn't via the
    # wrapper, so I'm using the RF importances for interpretation.
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'Feature':    features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\nFeature Importances:\n{imp.to_string(index=False)}",
              flush=True)
        imp.to_csv(
            os.path.join(output_dir, f"{slug}_importances.csv"), index=False
        )
        plt.figure(figsize=(8, 5))
        plt.barh(imp['Feature'], imp['Importance'])
        plt.title(f"Feature Importances -- {name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{slug}_importances.png"))
        plt.close()

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_te, y_prob, ax=ax)
    ax.set_title(f"ROC Curve -- {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slug}_roc_curve.png"))
    plt.close()

    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_te, y_prob, ax=ax)
    ax.axhline(
        baseline_prauc, linestyle='--', color='gray',
        label=f'Naive baseline ({baseline_prauc:.3f})'
    )
    ax.legend()
    ax.set_title(f"Precision-Recall Curve -- {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{slug}_pr_curve.png"))
    plt.close()

    return roc, pr


# ================================================================
# MODEL 1: RANDOM FOREST
# ================================================================
# I chose Random Forest as the first model to try because it handles
# non-linear relationships without me needing to scale anything first,
# and it gives feature importances out of the box which is useful for
# checking whether the model is picking up on what I'd expect.
#
# I used class_weight='balanced_subsample' instead of 'balanced' because
# with bagging, each tree sees a different bootstrap sample. Applying one
# global weight across all trees doesn't account for that. balanced_subsample
# rebalances within each individual tree, which is a better fit for RF.
print("\n" + "="*60, flush=True)
print("RANDOM FOREST -- searching on subsample (n_iter=10)...", flush=True)
print("="*60, flush=True)

rf_params = {
    # Capped at 150 - 400 trees on 2.85M rows would take 30+ min to refit.
    # With only 6 features, 100-150 is more than enough.
    'n_estimators':      [100, 150],
    # None = fully grown trees. Depth 10/20 adds regularization.
    # I let the search explore both to see what the data prefers.
    'max_depth':         [10, 20, None],
    # Higher values here mean the model needs more evidence before splitting,
    # which reduces overfitting on the minority class.
    'min_samples_split': randint(2, 15),
    'min_samples_leaf':  randint(1, 8),
    # Limits features per split so individual trees aren't too correlated
    # with each other that's what makes the ensemble useful.
    'max_features':      ['sqrt', 'log2'],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=rf_params,
    n_iter=10,                    # 10 combinations x 5 folds = 50 fits
    scoring='average_precision',  # PR-AUC as primary metric, same as baseline
    cv=cv5,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=False                   # refitting manually on full training set below
)

t0 = time.time()
rf_search.fit(X_tune, y_tune)
print(f"  Search done in {(time.time()-t0)/60:.1f} min", flush=True)
print(f"  Best params: {rf_search.best_params_}", flush=True)
print(f"  Best CV PR-AUC: {rf_search.best_score_:.4f}", flush=True)

pd.DataFrame(rf_search.cv_results_)[[
    'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
]].sort_values('rank_test_score').to_csv(
    os.path.join(output_dir, "rf_search_results.csv"), index=False
)

# I cap n_estimators at 150 here just to keep the full refit from running
# too long. The search found the optimal hyperparameters, this just limits
# how many trees go into the final model.
best_rf_params = rf_search.best_params_.copy()
best_rf_params['n_estimators'] = min(best_rf_params['n_estimators'], 150)

print(f"\nRefitting RF on full training set "
      f"(n_estimators={best_rf_params['n_estimators']})...", flush=True)
t0 = time.time()
best_rf = RandomForestClassifier(
    **best_rf_params,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
best_rf.fit(X_train, y_train)
print(f"  RF refit done in {(time.time()-t0)/60:.1f} min", flush=True)

rf_roc, rf_pr = evaluate(
    "Random Forest", best_rf, X_test, y_test,
    baseline_prauc, output_dir, features
)


# ================================================================
# MODEL 2: GRADIENT BOOSTING (HistGradientBoosting)
# ================================================================
# I went with Gradient Boosting as the second model because it builds
# trees sequentially and corrects its own errors as it goes, which tends
# to work well on imbalanced tabular data. Each round focuses on the
# examples the previous round got wrong, which helps with the minority class.
#
# I used HistGradientBoostingClassifier instead of the standard
# GradientBoostingClassifier mainly because it's much faster on large
# datasets. The histogram binning makes a big difference at 2.85M rows.
# The difference was roughly 2 minutes vs 20 minutes per fit.


class BalancedHGB(BaseEstimator, ClassifierMixin):
    """
    A wrapper around HistGradientBoostingClassifier that handles class
    balancing inside fit() rather than through fit_params.

    I had to write this because when I passed sample_weight through
    fit_params to RandomizedSearchCV, sklearn was applying those weights
    during scoring too, not just during training. That inflated CV PR-AUC
    from ~0.15 to ~0.78 because the scorer was seeing weighted metrics instead
    of the real class distribution. Moving the weight computation inside
    fit() means the CV scorer always evaluates on the unweighted distribution,
    which gives honest CV estimates.

    The weights themselves are simple: non-burdened households get weight 1.0,
    rent-burdened households get weight = n_negative / n_positive. That's
    equivalent to class_weight='balanced' in sklearn's linear models.
    """

    def __init__(self, max_iter=100, max_depth=None, learning_rate=0.1,
                 min_samples_leaf=20, l2_regularization=0.0,
                 max_leaf_nodes=31, random_state=42):
        self.max_iter          = max_iter
        self.max_depth         = max_depth
        self.learning_rate     = learning_rate
        self.min_samples_leaf  = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_leaf_nodes    = max_leaf_nodes
        self.random_state      = random_state

    def fit(self, X, y):
        # Compute weights here, inside fit(), so they never reach the scorer.
        # This is the whole point of the wrapper.
        neg, pos = (y == 0).sum(), (y == 1).sum()
        w = np.where(y == 1, neg / pos, 1.0)
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
        )
        self.model_.fit(X, y, sample_weight=w)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


print("\n" + "="*60, flush=True)
print("GRADIENT BOOSTING -- searching on subsample (n_iter=10)...",
      flush=True)
print("="*60, flush=True)
print("  (weights computed inside fit() -- no leakage into CV scorer)",
      flush=True)

gb_params = {
    # More rounds = more residual corrections, but more overfitting risk.
    # I let the search try 100/200/300 to see what pairs best with the
    # learning rate.
    'max_iter':          [100, 200, 300],
    # Shallow trees reduce variance; boosting corrects the bias across
    # rounds. I searched depth 3-6 to keep individual trees simple.
    'max_depth':         randint(3, 7),
    # Lower learning rate = smaller steps, needs more rounds to converge.
    # Higher = faster convergence but risks overshooting.
    'learning_rate':     uniform(0.02, 0.18),
    # One of the most important regularizers for imbalanced data -- stops
    # individual leaves from being driven by just a handful of minority
    # class examples.
    'min_samples_leaf':  randint(10, 80),
    # L2 penalty on leaf weights, extra smoothing on top of min_samples_leaf.
    'l2_regularization': uniform(0, 0.8),
    # Alternative to max_depth for capping tree complexity.
    'max_leaf_nodes':    [31, 63, None],
}

gb_search = RandomizedSearchCV(
    BalancedHGB(random_state=42),
    param_distributions=gb_params,
    n_iter=10,
    scoring='average_precision',  # unweighted, evaluates on real distribution
    cv=cv5,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=False
)

t0 = time.time()
gb_search.fit(X_tune, y_tune)
print(f"  Search done in {(time.time() - t0) / 60:.1f} min", flush=True)
print(f"  Best params: {gb_search.best_params_}", flush=True)
print(f"  Best CV PR-AUC: {gb_search.best_score_:.4f}", flush=True)

pd.DataFrame(gb_search.cv_results_)[[
    'params', 'mean_test_score', 'std_test_score', 'rank_test_score'
]].sort_values('rank_test_score').to_csv(
    os.path.join(output_dir, "gb_search_results.csv"), index=False
)

print(f"\nRefitting GB on full training set...", flush=True)
t0 = time.time()
best_gb = BalancedHGB(**gb_search.best_params_, random_state=42)
best_gb.fit(X_train, y_train)
print(f"  GB refit done in {(time.time() - t0) / 60:.1f} min", flush=True)

gb_roc, gb_pr = evaluate(
    "Gradient Boosting", best_gb, X_test, y_test,
    baseline_prauc, output_dir, features
)


# ================================================================
# COMPARISON TABLE
# ================================================================
# GB came out slightly ahead with PR-AUC 0.148 vs 0.146 for RF. I went with
# GB as the final model for two reasons: it had the higher PR-AUC on the
# minority class which is what I care about most, and the sequential
# correction approach is theoretically better suited for imbalanced problems
# than averaging independent trees. The 0.002 margin is small but with
# ~26K burdened households in the test set it translates to real families.
print("\n" + "="*60, flush=True)
print("MODEL COMPARISON", flush=True)
print("="*60, flush=True)

comp = pd.DataFrame({
    'Model': [
        'Logistic Regression (Baseline)',
        'Random Forest',
        'Gradient Boosting'
    ],
    'ROC-AUC':              [lr_roc,  rf_roc,  gb_roc],
    'PR-AUC':               [lr_pr,   rf_pr,   gb_pr],
    'PR-AUC lift vs naive': [
        lr_pr / baseline_prauc,
        rf_pr / baseline_prauc,
        gb_pr / baseline_prauc
    ],
})
print(comp.to_string(index=False), flush=True)
comp.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

x, w = np.arange(3), 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, comp['ROC-AUC'], w, label='ROC-AUC')
ax.bar(x + w/2, comp['PR-AUC'],  w, label='PR-AUC')
ax.axhline(
    baseline_prauc, linestyle='--', color='gray',
    label=f'Naive PR-AUC ({baseline_prauc:.3f})'
)
ax.set_xticks(x)
ax.set_xticklabels(comp['Model'], rotation=10, ha='right')
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: ROC-AUC and PR-AUC")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_comparison.png"))
plt.close()


# ================================================================
# FAIRNESS ANALYSIS
# ================================================================
# I ran fairness analysis on the best GB model. The three metrics I
# computed per subgroup are:
#   - PPR (Positive Prediction Rate): what share of each group gets flagged?
#     Gaps here mean unequal screening rates across groups.
#   - TPR (True Positive Rate): are burdened households caught at the same
#     rate across groups? Low TPR for a group means genuinely burdened
#     households there are being missed.
#   - FPR (False Positive Rate): are non-burdened households falsely flagged
#     at the same rate? High FPR for a group means that group bears a
#     disproportionate burden of false alarms.
#
# I used White and Male as reference groups because they're the majority
# groups in the test set and the conventional baseline for housing
# discrimination research.
print("\n" + "="*60, flush=True)
print("FAIRNESS ANALYSIS -- Gradient Boosting (best model)", flush=True)
print("="*60, flush=True)

race_map = {
    1: 'White', 2: 'Black', 3: 'American Indian',
    4: 'Chinese', 5: 'Japanese', 6: 'Other Asian/PI',
    7: 'Other', 8: 'Two+ Races', 9: 'Three+ Races',
}
sex_map = {1: 'Male', 2: 'Female'}

# I use df_model.loc[X_test.index] to make sure predictions and demographic
# labels are aligned since X_test is a subset of df_model, using .loc
# with the index is the safest way to guarantee they line up correctly.
df_test = df_model.loc[X_test.index].copy()
df_test['y_true']     = y_test.values
df_test['y_pred']     = best_gb.predict(X_test)
df_test['y_prob']     = best_gb.predict_proba(X_test)[:, 1]
df_test['RACE_label'] = df_test['RACE'].map(race_map).fillna('Unknown')
df_test['SEX_label']  = df_test['SEX'].map(sex_map).fillna('Unknown')

# Overall metrics: these go on the fairness plots as reference lines
# so I can see how each subgroup compares to the model overall.
overall_prauc = average_precision_score(df_test['y_true'], df_test['y_prob'])
overall_ppr   = df_test['y_pred'].mean()
overall_tpr   = (
    ((df_test['y_pred'] == 1) & (df_test['y_true'] == 1)).sum() /
    (df_test['y_true'] == 1).sum()
)
overall_fpr   = (
    ((df_test['y_pred'] == 1) & (df_test['y_true'] == 0)).sum() /
    (df_test['y_true'] == 0).sum()
)


def subgroup_metrics(df_sub, group_col, attribute_label, min_n=200):
    """
    Computes PPR, TPR, FPR, and PR-AUC for each subgroup within a
    demographic attribute.

    I set min_n=200 to exclude very small groups. Metric estimates for
    groups with fewer than 200 observations aren't reliable enough to
    interpret, so it's better to just drop them.

    Args:
        df_sub (pd.DataFrame): Test set with y_true, y_pred, y_prob columns.
        group_col (str):       Column with subgroup labels (e.g., 'RACE_label').
        attribute_label (str): Attribute name for the output (e.g., 'RACE').
        min_n (int):           Minimum group size to include (default: 200).

    Returns:
        pd.DataFrame: Per-group metrics sorted by group name.
    """
    rows = []
    for group, gdf in df_sub.groupby(group_col):
        if len(gdf) < min_n:
            continue
        yt = gdf['y_true'].values
        yp = gdf['y_pred'].values
        yb = gdf['y_prob'].values
        tp = ((yp == 1) & (yt == 1)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        tn = ((yp == 0) & (yt == 0)).sum()
        rows.append({
            'Group':      group,
            'n':          len(gdf),
            'Prevalence': yt.mean(),
            'PPR':        yp.mean(),
            'TPR':        tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            'FPR':        fp / (fp + tn) if (fp + tn) > 0 else np.nan,
            'PR_AUC':     (average_precision_score(yt, yb)
                           if yt.sum() > 0 else np.nan),
        })
    result = pd.DataFrame(rows).sort_values('Group')
    result.insert(0, 'Attribute', attribute_label)
    return result


def disparity_summary(metrics_df, ref_group, attribute_name):
    """
    Computes disparity gaps relative to a reference group.

    DP_diff and EO_FPR_diff are the most important outputs here. DP_diff
    shows how much more (or less) each group is flagged compared to the
    reference, and EO_FPR_diff shows how much worse the false positive rate
    is. The Black/White FPR gap ended up being the biggest finding from
    this analysis.

    Args:
        metrics_df (pd.DataFrame): Output of subgroup_metrics().
        ref_group (str):           Reference group (e.g., 'White', 'Male').
        attribute_name (str):      Attribute label for output column.

    Returns:
        pd.DataFrame: Metrics plus DP_diff, DP_ratio, EO_TPR_diff, EO_FPR_diff.
    """
    ref = metrics_df[metrics_df['Group'] == ref_group]
    if ref.empty:
        print(f"  Reference group '{ref_group}' not found.", flush=True)
        return pd.DataFrame()
    ref_ppr = ref['PPR'].values[0]
    ref_tpr = ref['TPR'].values[0]
    ref_fpr = ref['FPR'].values[0]
    rows = []
    for _, row in metrics_df.iterrows():
        rows.append({
            'Attribute':   attribute_name,
            'Group':       row['Group'],
            'n':           row['n'],
            'Prevalence':  row['Prevalence'],
            'PPR':         row['PPR'],
            'TPR':         row['TPR'],
            'FPR':         row['FPR'],
            'PR_AUC':      row['PR_AUC'],
            'DP_diff':     row['PPR'] - ref_ppr,
            'DP_ratio':    row['PPR'] / ref_ppr if ref_ppr > 0 else np.nan,
            'EO_TPR_diff': row['TPR'] - ref_tpr,
            'EO_FPR_diff': row['FPR'] - ref_fpr,
        })
    return pd.DataFrame(rows)


race_metrics = subgroup_metrics(df_test, 'RACE_label', 'RACE')
sex_metrics  = subgroup_metrics(df_test, 'SEX_label',  'SEX')

print("\n--- RACE FAIRNESS METRICS ---", flush=True)
print(race_metrics.to_string(index=False), flush=True)
print("\n--- SEX FAIRNESS METRICS ---", flush=True)
print(sex_metrics.to_string(index=False), flush=True)

pd.concat([race_metrics, sex_metrics], ignore_index=True).to_csv(
    os.path.join(output_dir, "fairness_metrics.csv"), index=False
)

race_disparity = disparity_summary(
    race_metrics, ref_group='White', attribute_name='RACE'
)
sex_disparity  = disparity_summary(
    sex_metrics, ref_group='Male', attribute_name='SEX'
)

print("\n--- DISPARITY SUMMARY (ref: White / Male) ---", flush=True)
for disp, name in [(race_disparity, 'RACE'), (sex_disparity, 'SEX')]:
    print(f"\n  {name}", flush=True)
    print(disp[[
        'Group', 'n', 'PPR', 'DP_diff', 'DP_ratio',
        'TPR', 'EO_TPR_diff', 'FPR', 'EO_FPR_diff'
    ]].to_string(index=False), flush=True)

pd.concat([race_disparity, sex_disparity], ignore_index=True).to_csv(
    os.path.join(output_dir, "fairness_disparity.csv"), index=False
)


# -----------------------------
# Fairness Plots
# -----------------------------
def bar_chart(metrics_df, metric, title, ylabel, filename,
              overall_val=None, overall_label=None, pct=False):
    """
    Bar chart of a fairness metric by subgroup. Reference groups (White,
    Male) are colored red; all others are blue. The dashed line shows
    the overall model value so you can see whether each group is above
    or below average.

    Args:
        metrics_df (pd.DataFrame): Subgroup metrics table.
        metric (str):              Column to plot ('PPR', 'TPR', 'FPR').
        title (str):               Chart title.
        ylabel (str):              Y-axis label.
        filename (str):            Output file name.
        overall_val (float):       Overall reference line value.
        overall_label (str):       Label for the reference line.
        pct (bool):                Format y-axis as percentages if True.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        '#d73027' if g in ('White', 'Male') else '#4575b4'
        for g in metrics_df['Group']
    ]
    ax.bar(metrics_df['Group'], metrics_df[metric],
           color=colors, edgecolor='white')
    if overall_val is not None:
        ax.axhline(
            overall_val, linestyle='--', color='black', linewidth=1.0,
            label=f'Overall ({overall_label}: {overall_val:.3f})'
        )
        ax.legend(fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    if pct:
        ax.yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1, decimals=0)
        )
    ax.set_ylim(0, metrics_df[metric].max() * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


print("\nGenerating fairness plots...", flush=True)

for grp, metrics in [('race', race_metrics), ('sex', sex_metrics)]:
    bar_chart(
        metrics, 'PPR',
        f'Positive Prediction Rate by {grp.title()} (Demographic Parity)',
        'Predicted Positive Rate', f'fairness_{grp}_ppr.png',
        overall_val=overall_ppr, overall_label='overall PPR', pct=True
    )
    bar_chart(
        metrics, 'TPR',
        f'True Positive Rate by {grp.title()} (Equalized Odds -- Opportunity)',
        'True Positive Rate (Recall)', f'fairness_{grp}_tpr.png',
        overall_val=overall_tpr, overall_label='overall TPR', pct=True
    )
    bar_chart(
        metrics, 'FPR',
        f'False Positive Rate by {grp.title()} (Equalized Odds -- Harm)',
        'False Positive Rate', f'fairness_{grp}_fpr.png',
        overall_val=overall_fpr, overall_label='overall FPR', pct=True
    )
    bar_chart(
        metrics, 'PR_AUC',
        f'PR-AUC by {grp.title()}',
        'PR-AUC', f'fairness_{grp}_prauc.png',
        overall_val=overall_prauc, overall_label='overall PR-AUC'
    )

# Diverging bar chart: shows direction and magnitude of disparities
# relative to White. Red bars = over-flagged or higher TPR than White,
# blue = under-flagged or lower TPR. The Black/White FPR gap was the
# biggest finding: non-burdened Black households were flagged as positive
# at a rate 22+ percentage points higher than non-burdened White households.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, label in zip(
    axes,
    ['DP_diff', 'EO_TPR_diff'],
    ['Demographic Parity Difference\n(PPR minus White PPR)',
     'Equalized Odds Difference\n(TPR minus White TPR)']
):
    sub    = race_disparity[race_disparity['Group'] != 'Unknown'].copy()
    colors = ['#d73027' if v > 0 else '#4575b4' for v in sub[col]]
    ax.barh(sub['Group'], sub[col], color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Difference from White reference group')
    ax.xaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=1, decimals=1)
    )
plt.suptitle(
    'Racial Disparity in Model Predictions (Reference: White)',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "fairness_race_disparity.png"),
    dpi=150, bbox_inches='tight'
)
plt.close()

print(f"\nAll outputs saved to: {output_dir}", flush=True)
print("\nAdvanced models + fairness analysis complete.", flush=True)

# -----------------------------
# Testing & Validation
# -----------------------------

"""
SELF TEST: AH on Local Machine
------------------------------
- Script runs end-to-end without errors in PyCharm
- Same preprocessing as baseline_model.py confirmed positive class
  rate 3.65%, row count 3.56M, feature set matches
- BalancedHGB wrapper confirmed working: GB CV PR-AUC ~0.135 on subsample
  (plausible), vs ~0.78 before the fix (clearly inflated)
- Tuning subsample positive rate = 3.65%, matches full training set
- RF refit on full training set: test PR-AUC = 0.146
- GB refit on full training set: test PR-AUC = 0.148
- Gradient Boosting selected as best model
- df_model.loc[X_test.index] alignment confirmed correct for fairness analysis
- Race and sex subgroup metrics computed without errors (groups < 200 excluded)
- Key finding: Black/White FPR gap = ~22 percentage points
- All CSVs and PNGs saved without errors

USER TEST: Secondary Device
---------------------------
- Run baseline_model.py first so baseline_metrics.csv exists
- Script is portable after path update
- Raw IPUMS data not in repo, download from: https://usa.ipums.org/usa/

REQUIRED USER STEPS:
1. Run baseline_model.py first
2. Download 2024 ACS 5-Year dataset from IPUMS: https://usa.ipums.org/usa/
3. Update `path` and `output_dir`
4. Install: pandas, numpy, matplotlib, scikit-learn, scipy

USER TEST STATUS: PASSED (with dataset dependency noted)
"""