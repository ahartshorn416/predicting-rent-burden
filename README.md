# 🏠 Predicting Rent Burden in U.S. Households Using Machine Learning & Fairness Analysis

## 📌 Overview
This project analyzes whether U.S. households are rent burdened — defined as spending more than 30% of household income on rent — using large-scale survey data and machine learning techniques. In addition to predicting rent burden status, the project evaluates how model performance varies across demographic groups to assess fairness and equity.

The goal is to generate insights that can support policymakers in identifying at-risk households and improving housing affordability strategies.

---

## ❓ Research Question
Can machine learning models predict which U.S. households are rent burdened, and does model performance vary across demographic and geographic groups in ways that inform housing policy decisions?

---

## 🎯 Objectives
- Predict rent burden status using machine learning models
- Identify key drivers of housing affordability
- Evaluate model performance across demographic groups
- Assess fairness using multiple metrics
- Provide policy-relevant insights

---

## 📊 Data Source
- **Dataset:** 2024 American Community Survey (ACS) 5-Year PUMS
- **Access:** IPUMS USA
- **Format:** `.csv` file processed into a structured dataset using Python
- **Size:** ~16 million observations (restricted to ~3.56M renter-only households for modeling)

---

## 🧾 Selected Variables
```
YEAR, MULTYEAR, SAMPLE, SERIAL, CBSERIAL,
STATEFIP, PUMA, OWNERSHP, RENTGRS, HHINCOME,
ROOMS, BEDROOMS, SEX, AGE, RACE, EDUC,
EMPSTAT, WKSWORK1, OCC
```

---

## 🧠 Target Variable
```python
rent_burdened = 1 if (RENTGRS / HHINCOME) > 0.3 else 0
# Households with HHINCOME <= 0 are assigned rent_burdened = 1 by definition
```

---

## 🛠️ Custom Functions

### `preprocessing.py` (shared module — repo root)
All four modeling scripts import from this shared module. It contains the canonical definitions for every preprocessing and feature engineering decision, so changes only need to be made in one place.

| Function | Arguments | Returns | Purpose |
|----------|-----------|---------|---------|
| `load_and_clean(path, cols_needed)` | CSV path, list of columns | Cleaned renter-only DataFrame | Replaces IPUMS sentinel, median imputes, filters to renters |
| `build_features(df)` | Cleaned DataFrame | DataFrame with engineered columns | Creates target variable, log transforms, rent-to-income ratio, UNSTABLE_EMPLOYMENT |
| `get_feature_list()` | None | List of 6 feature names | Returns the canonical model feature set |
| `get_target()` | None | String `'rent_burdened'` | Returns the target column name |

**Usage example:**
```python
from preprocessing import load_and_clean, build_features, get_feature_list, get_target

df = load_and_clean(path, cols_needed=['HHINCOME', 'RENTGRS', ...])
df = build_features(df)
features = get_feature_list()  # ['AGE', 'EDUC', 'SEX', 'RACE', 'WKSWORK1', 'UNSTABLE_EMPLOYMENT']
target   = get_target()        # 'rent_burdened'
```

### `BalancedHGB` (in `advanced_models.py`)
A custom wrapper around `HistGradientBoostingClassifier` that computes balanced sample weights internally during `fit()`. This prevents weight leakage into the `RandomizedSearchCV` scorer, which was inflating CV PR-AUC from ~0.15 to ~0.78 when weights were passed via `fit_params`.

```python
model = BalancedHGB(max_iter=100, max_depth=5, learning_rate=0.191,
                    min_samples_leaf=33, l2_regularization=0.300,
                    max_leaf_nodes=31, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

### `evaluate(name, model, X_te, y_te, ...)` (in `advanced_models.py`)
Evaluates a fitted model on the test set and saves all outputs (classification report, confusion matrix, feature importances if available, ROC curve, PR curve). Returns `(roc_auc, pr_auc)` for the comparison table.

### `subgroup_metrics(df_sub, group_col, attribute_label, min_n)` (in `advanced_models.py`)
Computes PPR, TPR, FPR, and PR-AUC for each demographic subgroup. Groups with fewer than `min_n` observations (default 200) are excluded to avoid unreliable metric estimates.

### `disparity_summary(metrics_df, ref_group, attribute_name)` (in `advanced_models.py`)
Computes disparity gaps (DP_diff, DP_ratio, EO_TPR_diff, EO_FPR_diff) relative to a reference group (White or Male).

---

## 🛠️ Feature Engineering
- Created **rent-to-income ratio** (excluded from model feature set to prevent target leakage)
- Generated **UNSTABLE_EMPLOYMENT** indicator:
  - Flagged households with `WKSWORK1 < 35` or `EMPSTAT ∈ {2, 3}` (unemployed / not in labor force)
- Applied **log transformations** to reduce skew on income and rent variables
- Restricted dataset to **renter-only households** (`RENTGRS > 0`) — non-renters cannot be rent-burdened by definition
- Handled IPUMS sentinel value 9,999,999 for missing income (replaced with NaN, then median imputed)

---

## ⚙️ Preprocessing
- Replaced IPUMS sentinel income values with NaN; applied median imputation (~5.5% of full dataset affected, much smaller proportion within renter-only subset)
- Removed non-renter and zero/negative income observations
- Split data into **train (80%) / test (20%)** with stratification on target variable
- `StandardScaler` and `PCA` fit on training data only and applied to test set to prevent data leakage
- Income, rent, and ratio features (`log_income`, `log_rent`, `rent_income_ratio`) excluded from model feature set after initial run produced perfect ROC-AUC = 1.0 (target leakage confirmed)

---

## 📈 Exploratory Data Analysis (EDA)
- Generated summary tables for categorical and continuous variables
- Assessed skewness and kurtosis: HHINCOME skew = 4.31, RENTGRS skew = 2.03 (renter-only subset)
- Rent burden rate varied across years: 3.0% in 2020, peaking at 3.9% in 2021, stabilizing at 3.6% in 2024
- Visualizations included:
  - Histograms (income, rent)
  - Boxplots (income by rent burden status)
  - Correlation matrix
  - Pairplots (sampled to 50K rows for performance)
  - Rent burden rate by year

---

## 🤖 Models

### ✅ Baseline: Logistic Regression

The baseline logistic regression model was trained on 6 demographic and employment features: `AGE`, `EDUC`, `SEX`, `RACE`, `WKSWORK1`, and `UNSTABLE_EMPLOYMENT`. Income and rent variables were excluded to prevent target leakage. `class_weight='balanced'` was used to address the 3.65% positive class rate without resampling.

#### Evaluation Metrics (Test Set, n = 712,928)

| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | 0.773 | Solid discriminative ability for a baseline |
| PR-AUC | 0.097 | ~2.6x above naive baseline (0.037) |
| Naive Baseline PR-AUC | 0.037 | Positive class rate in training set |
| Accuracy | 0.54 | Low due to `class_weight='balanced'` tradeoff |
| Precision (class 1) | 0.07 | Many false positives; wide net cast |
| Recall (class 1) | 0.92 | 92% of rent-burdened households identified |
| F1-Score (class 1) | 0.13 | Reflects precision-recall tradeoff |

#### Top Predictors (Logistic Regression Coefficients)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| WKSWORK1 | -1.288 | Strongest predictor: more weeks worked strongly reduces risk |
| UNSTABLE_EMPLOYMENT | +0.357 | Unstable work increases predicted burden — consistent with hypothesis |
| EDUC | +0.331 | Positive direction unexpected; may reflect urban residence confound |
| AGE | -0.091 | Older individuals slightly less likely to be flagged |
| SEX | +0.048 | Small positive effect |
| RACE | -0.033 | Small negative effect; treat with caution (ordinal coding limitation) |

#### Cross-Validation Results (5-Fold Stratified)

| Fold | Train ROC-AUC | Val ROC-AUC | Train PR-AUC | Val PR-AUC |
|------|--------------|-------------|--------------|------------|
| 1 | 0.7728 | 0.7714 | 0.0968 | 0.0968 |
| 2 | 0.7726 | 0.7724 | 0.0968 | 0.0965 |
| 3 | 0.7727 | 0.7721 | 0.0967 | 0.0969 |
| 4 | 0.7719 | 0.7752 | 0.0969 | 0.0981 |
| 5 | 0.7728 | 0.7715 | 0.0971 | 0.0961 |
| **Mean (Val)** | — | **0.7725 ± 0.0016** | — | **0.0969 ± 0.0008** |

Training and validation scores are nearly identical across all folds, indicating no evidence of overfitting.

---

### ✅ Advanced Models: Random Forest & Gradient Boosting

Both models used the same leakage-free 6-feature set as the baseline. Hyperparameter tuning was performed via `RandomizedSearchCV` with 5-fold stratified cross-validation, optimizing for `average_precision` (PR-AUC). To avoid timeout on the 2.85M-row training set, search was conducted on a stratified 5% subsample (~142,585 rows); best parameters were then refit on the full training set.

#### Hyperparameter Tuning — Random Forest

`class_weight='balanced_subsample'` was used (per-tree balancing; more appropriate than global reweighting for bagging ensembles).

| Hyperparameter | Best Value | Search Range |
|----------------|-----------|--------------|
| n_estimators | 100 | [100, 150] |
| max_depth | 10 | [10, 20, None] |
| min_samples_split | 11 | randint(2, 15) |
| min_samples_leaf | 7 | randint(1, 8) |
| max_features | sqrt | ['sqrt', 'log2'] |

#### Hyperparameter Tuning — Gradient Boosting

`HistGradientBoostingClassifier` was used for its speed on large datasets. A custom `BalancedHGB` wrapper computed balanced sample weights internally during `fit()` to prevent weight leakage into the CV scorer (passing `sample_weight` via `fit_params` inflated CV PR-AUC from ~0.15 to ~0.78).

| Hyperparameter | Best Value | Search Range |
|----------------|-----------|--------------|
| max_iter | 100 | [100, 200, 300] |
| max_depth | 5 | randint(3, 7) |
| learning_rate | 0.191 | uniform(0.02, 0.18) |
| min_samples_leaf | 33 | randint(10, 80) |
| l2_regularization | 0.300 | uniform(0, 0.8) |
| max_leaf_nodes | 31 | [31, 63, None] |

#### Model Comparison (Test Set, n = 712,928)

| Model | ROC-AUC | PR-AUC | PR-AUC Lift vs. Naive |
|-------|---------|--------|-----------------------|
| Logistic Regression (Baseline) | 0.773 | 0.097 | 2.65x |
| Random Forest | 0.809 | 0.146 | 4.00x |
| **Gradient Boosting** ✅ | **0.810** | **0.148** | **4.06x** |

**Gradient Boosting was selected as the best model.** It achieves the highest PR-AUC on the minority class (the primary evaluation criterion) and its sequential residual correction is theoretically better suited than bagging for imbalanced classification.

#### Feature Importances (Random Forest)

| Feature | Importance |
|---------|-----------|
| WKSWORK1 | 0.494 |
| UNSTABLE_EMPLOYMENT | 0.342 |
| AGE | 0.093 |
| RACE | 0.034 |
| EDUC | 0.033 |
| SEX | 0.004 |

`WKSWORK1` and `UNSTABLE_EMPLOYMENT` together account for 83.6% of model decisions, strongly consistent with the hypothesis that labor market instability is the primary driver of rent burden.

---

## ⚖️ Fairness Analysis

Fairness analysis was conducted on the best Gradient Boosting model using the held-out test set. Three metrics were computed per subgroup:
- **PPR** (Positive Prediction Rate) — demographic parity proxy
- **TPR** (True Positive Rate / Recall) — equalized odds, opportunity
- **FPR** (False Positive Rate) — equalized odds, harm

Reference groups: White (RACE), Male (SEX).

#### Race Subgroup Results

| Group | n | Prevalence | PPR | TPR | FPR | PR-AUC |
|-------|---|-----------|-----|-----|-----|--------|
| White (ref) | 364,374 | 3.1% | 31.9% | 77.1% | 30.5% | 0.127 |
| Black | 106,283 | 5.9% | 55.7% | 95.0% | 53.2% | 0.162 |
| American Indian | 12,056 | 3.8% | 47.8% | 87.0% | 46.2% | 0.102 |
| Chinese | 9,695 | 7.7% | 47.3% | 95.3% | 43.4% | 0.431 |
| Other Asian/PI | 38,482 | 3.4% | 25.8% | 73.5% | 24.1% | 0.184 |
| Other | 74,381 | 3.2% | 44.5% | 84.5% | 43.2% | 0.091 |
| Two+ Races | 97,551 | 3.1% | 43.1% | 81.4% | 41.9% | 0.097 |
| Three+ Races | 8,172 | 3.2% | 43.7% | 80.2% | 42.5% | 0.091 |
| Japanese | 1,934 | 3.4% | 27.5% | 83.1% | 25.5% | 0.187 |

#### Sex Subgroup Results

| Group | n | Prevalence | PPR | TPR | FPR | PR-AUC |
|-------|---|-----------|-----|-----|-----|--------|
| Male (ref) | 339,571 | 3.2% | 34.1% | 81.0% | 32.5% | 0.149 |
| Female | 373,357 | 4.0% | 42.6% | 84.7% | 40.9% | 0.147 |

#### Key Fairness Findings
- **Black households** show the largest disparity: PPR of 55.7% vs. 31.9% for White (DP ratio 1.74x), with an FPR gap of +22.7pp
- **Other Asian/PI households** are under-flagged: negative DP_diff (-0.061) and EO_TPR_diff (-0.035) indicate genuinely burdened households are more likely to be missed
- **Female households** are over-flagged relative to Male (FPR gap +8.3pp), partially explained by higher true prevalence (4.0% vs. 3.2%)
- **Chinese households** have the highest PR-AUC (0.431), likely due to higher true prevalence (7.7%)

---

## 📊 Evaluation Metrics
- **Precision-Recall AUC** (primary metric due to class imbalance)
- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Demographic Parity (PPR difference and ratio)
- Equalized Odds (TPR and FPR differences)

**Positive Class Rate (renter-only dataset):** 3.65%
**Naive Baseline PR-AUC:** 0.0365

---

## 📌 Key Findings

### EDA Stage
- Rent burden is rare (~3.65% among renters), indicating strong class imbalance
- Income and rent variables are highly skewed on the renter-only subset (HHINCOME skew = 4.31, RENTGRS skew = 2.03), requiring log transformation
- ~54% of renter households show unstable employment
- Non-renter filtering reduced dataset from 16.1M to 3.6M rows
- Rent burden rates were relatively stable across survey years (3.0%–3.9%), supporting treatment of pooled data as a single cross-section

### Baseline Model Stage
- Logistic regression achieves ROC-AUC of 0.773 and PR-AUC of 0.097 using only demographic and employment features
- Weeks worked (`WKSWORK1`) is the strongest predictor (coefficient = -1.288)
- Unstable employment is the strongest positive predictor (coefficient = +0.357), consistent with the hypothesis
- No overfitting detected (mean Val ROC-AUC = 0.773, SD = 0.002 across 5 folds)

### Advanced Models Stage
- Both tree-based models substantially outperform the baseline (~53% relative PR-AUC gain)
- Gradient Boosting selected as best model (ROC-AUC: 0.810, PR-AUC: 0.148, 4.06x above naive)
- Labor market features dominate: WKSWORK1 and UNSTABLE_EMPLOYMENT account for 83.6% of RF feature importance

### Fairness Analysis Stage
- Largest equalized odds violation: Black/White FPR gap of 22.7 percentage points
- Other Asian/PI households are systematically under-flagged despite similar prevalence to White households
- Sex disparities exist but are partially explained by true prevalence differences

---

## ⚠️ Limitations
- Cross-sectional data (no time trends)
- Self-reported income and rent
- Severe class imbalance (3.65% positive rate)
- Observational data limits causal conclusions
- `RACE` treated as ordinal integer — one-hot encoding to be explored in final model phase
- Median imputation for `HHINCOME` may modestly compress income variance near the median
- Hyperparameter search conducted on 5% subsample due to dataset size; best params refit on full training set
- PCA retained 4 components (not 6 as in earlier reports) after leaky features were correctly removed from the feature set

---

## 👥 Stakeholder
- U.S. Department of Housing and Urban Development (HUD)

---

## 🚀 Next Steps
- Explore one-hot encoding of `RACE` to address ordinal treatment limitation
- Synthesize all phases into final deliverable with policy recommendations
- Investigate positive coefficient on `EDUC` — potential confound with urban residence

---

## 📚 Sources
- U.S. Census Bureau — American Community Survey (ACS)
- IPUMS USA — ACS Microdata
- HUD — Housing Reports
- Joint Center for Housing Studies (Harvard University)

---

## 🚀 How to Run This Project

### 📥 1. Clone or download the repository
```bash
git clone https://github.com/ahartshorn416/prediciting_rent_burden
cd prediciting_rent_burden
```

### 📊 2. Download the dataset (REQUIRED)

This project uses IPUMS ACS microdata, which is not included due to file size.

Steps:
1. Go to https://usa.ipums.org/usa/
2. Create a free account
3. Select the 2024 ACS 5-Year dataset and the variables listed above
4. Download as CSV format
5. Save it to your local machine

### ⚙️ 3. Update file paths in scripts

Open each Python file and update the dataset path and output directory:
```python
path       = r"C:\Users\YOUR_USERNAME\Downloads\your_file.csv"
output_dir = r"C:\Users\YOUR_USERNAME\...\results"
```

### 📦 4. Install required packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupytext
```

### 📁 5. Repository structure

```
predicting-rent-burden/
    preprocessing.py              ← shared module (REQUIRED — must stay at repo root)
    eda/
        eda.py
    feature engineering/
        feature_engineering.py
    baseline model/
        baseline_model.py
    advanced models/
        advanced_models.py
    results/                      ← all outputs saved here
```

> **Important:** `preprocessing.py` must remain at the repo root. All four scripts import from it using a path that looks one level up from their subfolder. Moving it will break the imports.

### ▶️ 6. Run the scripts in order
```bash
python eda/eda.py
python "feature engineering/feature_engineering.py"
python "baseline model/baseline_model.py"
python "advanced models/advanced_models.py"   # includes fairness analysis
```

> Run `baseline_model.py` before `advanced_models.py` — the advanced models script loads `baseline_metrics.csv` from the results folder for the comparison table.

### 📂 7. Output files

All outputs are saved in `/results`, including:

| File | Description |
|------|-------------|
| `baseline_metrics.csv` | Logistic regression evaluation metrics |
| `baseline_coefficients.csv` | Logistic regression coefficients |
| `baseline_cv_results.csv` | Cross-validation fold results |
| `rf_search_results.csv` | Random Forest hyperparameter search results |
| `gb_search_results.csv` | Gradient Boosting hyperparameter search results |
| `model_comparison.csv` | ROC-AUC and PR-AUC across all three models |
| `fairness_metrics.csv` | Per-subgroup PPR, TPR, FPR, PR-AUC |
| `fairness_disparity.csv` | Disparity measures vs. reference groups |
| `*.png` | ROC curves, PR curves, feature importance plots, fairness bar charts |
