"""
feature_engineering.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

This script builds the dataset I use for all the modeling scripts. The main
things it does are load and clean the ACS data, build the features and target
variable, split into train/test sets, scale the features, and run PCA.

A few decisions I made here that are worth flagging:
  - I originally included log_income, log_rent, and rent_income_ratio in the
    feature set, but the first run gave me ROC-AUC = 1.0 which was obviously
    wrong. Those features reconstruct the target directly, so I removed them.
  - I fit the scaler and PCA on the training set only. Fitting on the full
    dataset before splitting would leak test set information into the model.
  - PCA is just for exploration here. I'm not feeding PCA components into
    the actual models as they train on the original features.

Roles:
- Feature Engineering Function: AH
- Feature Engineering: AH
- Select Features: AH
- Train Test Split: AH
- Fit StandardScaler + PCA on training data: AH
- Testing & Validation: AH
"""

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# preprocessing.py lives at the repo root, one level up from this subfolder.
# This line tells Python where to find it so the import doesn't fail.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# I moved all the cleaning and feature engineering into preprocessing.py
# so every script uses the exact same definitions. Before I did this,
# this script had a 9-feature list that included the leaky variables while
# the modeling scripts had 6. They would have trained on different data
# without raising any errors.
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

# -----------------------------
# Load & Clean
# -----------------------------
# load_and_clean() handles the sentinel replacement, median imputation, and
# renter filter. I'm calling it here rather than repeating those steps.
# See preprocessing.py for why I made each decision and why the order matters.
cols_needed = [
    'MULTYEAR', 'HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT',
    'AGE', 'EDUC', 'SEX', 'RACE', 'OWNERSHP'
]
df = load_and_clean(path, cols_needed=cols_needed)

# -----------------------------
# Feature Engineering
# -----------------------------
# build_features() creates the target variable, log transforms, rent-to-income
# ratio, and UNSTABLE_EMPLOYMENT. Same function used in baseline_model.py and
# advanced_models.py, so everything is guaranteed to be consistent.
df = build_features(df)

print("\nTarget distribution:")
print(df["rent_burdened"].value_counts())
print(f"Positive class rate: {df['rent_burdened'].mean():.4f}")

print("\nEmployment instability distribution:")
print(df["UNSTABLE_EMPLOYMENT"].value_counts())

# -----------------------------
# Select Features
# -----------------------------
# get_feature_list() returns the 6 demographic and employment features I
# settled on after catching the leakage issue. I'm using the function
# instead of hardcoding the list here so if I ever change the feature
# set in preprocessing.py, it updates everywhere automatically.
features = get_feature_list()
target   = get_target()

# Quick check so the script fails loudly if a column is missing rather
# than crashing somewhere downstream with a confusing error.
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

df_model = df[features + [target]].dropna()
print("\nFinal modeling dataset shape:", df_model.shape)

# -----------------------------
# Train / Test Split
# -----------------------------
X = df_model[features]
y = df_model[target]

# I went with 80/20 because at 3.56M rows, even the 20% test set gives
# me ~712K observations which is way more than I need for stable metrics.
# stratify=y is important here without it, random variation could give
# the test set a different positive rate than the training set, which would
# make PR-AUC comparisons between models less reliable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# StandardScaler + PCA
# -----------------------------
# I fit the scaler on X_train only and then just transform X_test with
# the learned parameters. If I fit on the full dataset first, the test
# set's distribution would influence the scaling, which is leakage.
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# PCA here is just exploratory. I wanted to see how many independent
# dimensions are actually in the feature set. Setting n_components=0.90
# means I keep however many components it takes to explain 90% of the
# variance. It came back with 4 components, which made sense given that
# income, employment, and demographics are all correlated with each other.
#
# I'm not using the PCA output in the actual models. The logistic regression,
# random forest, and gradient boosting all train on the original features.
# This was just to get a sense of the structure of the data.
pca         = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

pd.DataFrame(X_train_pca).to_csv(
    os.path.join(output_dir, "X_train_pca.csv"), index=False
)
pd.DataFrame(X_test_pca).to_csv(
    os.path.join(output_dir, "X_test_pca.csv"), index=False
)

print("\nPCA components retained:", pca.n_components_)

# -----------------------------
# Export Files
# -----------------------------
df_model.to_csv(os.path.join(output_dir, "model_data.csv"), index=False)
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

print("\nAll files saved.")

# -----------------------------
# Testing & Validation
# -----------------------------

"""
SELF TEST: AH on Local Machine
------------------------------
- Script runs without errors in PyCharm
- Confirmed usecols is working -- no memory errors on the 16M row file
- IPUMS sentinel replaced before any computation -- checked via print in
  preprocessing.py that max(HHINCOME) drops from 9999999 after replacement
- Row count drops from 16.1M to 3.56M after renter filter, as expected
- Positive class rate: 3.65% matches EDA output exactly
- UNSTABLE_EMPLOYMENT: 54.4% flagged matches EDA
- Leakage check: log_income, log_rent, rent_income_ratio built by
  build_features() but not in get_feature_list() confirmed excluded
- stratify=y confirmed: positive rate in train and test both = 3.65%
- Scaler fit on X_train only (fit_transform), applied to X_test with
  transform only leakage prevention confirmed
- PCA retained 4 components at 90% variance threshold
- All CSVs saved to results folder without errors

USER TEST: Secondary Device
---------------------------
- Script is portable only the path variables need updating
- Raw IPUMS data not in repo due to file size
- Download from: https://usa.ipums.org/usa/

REQUIRED USER STEPS:
1. Download 2024 ACS 5-Year dataset from IPUMS (CSV format)
   Variables: YEAR, MULTYEAR, SAMPLE, SERIAL, CBSERIAL, STATEFIP, PUMA,
              OWNERSHP, RENTGRS, HHINCOME, ROOMS, BEDROOMS, SEX, AGE,
              RACE, EDUC, EMPSTAT, WKSWORK1, OCC
2. Update `path` to your local file location
3. Update `output_dir` to your results directory
4. Install: pandas, numpy, scikit-learn

USER TEST STATUS: PASSED (with dataset dependency noted)
"""