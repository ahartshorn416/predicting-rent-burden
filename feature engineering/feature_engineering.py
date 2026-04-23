"""
feature_engineering.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

This script builds a machine learning dataset to predict whether U.S. households are rent-burdened by
loading ACS data, engineering financial and employment features, builds target variable, splitting into train/test sets,
applying PCA, and exporting the processed datasets for modeling.

Roles:

- Feature Engineering Function: AH
- Feature Engineering: AH
- Select Features: AH
- Train Test Split: AH
- Fit StandardScaler + PCA on training data: AH
- Testing & Validation: AH
"""

#  -----------------------------
# Imports
#  -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

#  -----------------------------
# Path Settings
#  -----------------------------
path = r"C:\Users\alica\Downloads\usa_00003.csv"
output_dir = r"C:\Users\alica\OneDrive\Documents\predicting-rent-burden\results"

#  -----------------------------
# Load Data
#  -----------------------------
cols_needed = ['MULTYEAR', 'HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT',
               'AGE', 'EDUC', 'SEX', 'RACE', 'OWNERSHP']

df = pd.read_csv(path, usecols=cols_needed)
print(f"Data loaded: {df.shape}")

#  -----------------------------
# Pre-processing (consistent with EDA)
#  -----------------------------

# Replace IPUMS sentinel value for missing income with NaN, then median impute
df['HHINCOME'] = df['HHINCOME'].replace(9999999, np.nan)
df['HHINCOME'] = df['HHINCOME'].fillna(df['HHINCOME'].median())

# Restrict to renter households only (RENTGRS > 0)
# Non-renters are excluded because:
#   1. They cannot be rent-burdened by definition
#   2. Including them inflates the negative class and worsens class imbalance

n_before = len(df)
df = df[df['RENTGRS'] > 0].copy()
print(f"Restricted to renters: {n_before - len(df)} rows removed ({n_before} -> {len(df)})")

#  -----------------------------
#  Feature Engineering function
#  -----------------------------
def build_rent_burden_features(df):
    """
    Creates target + engineered features for rent burden prediction.
    """

    df = df.copy()

    # -------------------------
    # Target variable
    # -------------------------
    df["rent_burdened"] = np.where(
        df["HHINCOME"] <= 0,
        1,
        np.where((df["RENTGRS"] / df["HHINCOME"]) > 0.3, 1, 0)
    )

    # -------------------------
    # Log features
    # -------------------------
    df["log_income"] = np.log1p(df["HHINCOME"].clip(lower=0))
    df["log_rent"] = np.log1p(df["RENTGRS"].clip(lower=0))

    # -------------------------
    # Ratio feature
    # -------------------------
    df["rent_income_ratio"] = df["RENTGRS"] / (df["HHINCOME"] + 1)

    # -------------------------
    # Employment instability
    # -------------------------
    df["UNSTABLE_EMPLOYMENT"] = np.where(
        (df["WKSWORK1"] < 35) | (df["EMPSTAT"].isin([2, 3])),
        1, 0
    )

    return df

# -----------------------------
# Feature Engineering
# -----------------------------
df = build_rent_burden_features(df)

print("\nTarget distribution:")
print(df["rent_burdened"].value_counts())
print(f"Positive class rate: {df['rent_burdened'].mean():.4f}")

print("\nEmployment instability distribution:")
print(df["UNSTABLE_EMPLOYMENT"].value_counts())

#  -----------------------------
# Select Features
#  -----------------------------
features = [
    "log_income",
    "log_rent",
    "rent_income_ratio",
    "AGE",
    "EDUC",
    "SEX",
    "RACE",
    "WKSWORK1",
    "UNSTABLE_EMPLOYMENT"
]

target = "rent_burdened"

# Ensure no missing columns crash script
missing = [col for col in features + [target] if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

df_model = df[features + [target]].dropna()

print("\nFinal modeling dataset shape:", df_model.shape)

# -----------------------------
# Train/ Test Split
# -----------------------------
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# Fit StandardScaler + PCA on training data
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

pd.DataFrame(X_train_pca).to_csv(
    os.path.join(output_dir, "X_train_pca.csv"),
    index=False
)

pd.DataFrame(X_test_pca).to_csv(
    os.path.join(output_dir, "X_test_pca.csv"),
    index=False
)

print("\nPCA components retained:", pca.n_components_)

# -----------------------------
# Export Files
# -----------------------------
df_model.to_csv(os.path.join(output_dir, "model_data.csv"), index=False)
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

print("\nAll processed files saved successfully.")

# -----------------------------
# Testing & Validation
# -----------------------------

"""
SELF TEST: AH on Local Machine
------------------------------
- Script runs successfully on Pycharm
- No runtime errors encountered
- usecols used to avoid memory allocation error on large dataset
- IPUMS sentinel value (9999999) replaced with NaN before any processing
- Non-renter households filtered out before feature engineering
- Zero-income edge case handled consistently with EDA operationalization
- Engineered features were created correctly
- Train/Split successful
- PCA executed without errors
- Results saved successfull

USER TEST: Secondary Device 
___________________________
- Script is designed to be portable across machines
- Due to large file size (~IPUMS ACS dataset), raw data is NOT included in repository
- User must download dataset directly from IPUMS USA:
    https://usa.ipums.org/usa/

REQUIRED USER STEPS:
1. Download ACS dataset from IPUMS (CSV format)
    - 2024 5 year ACS
    - Selected Variables: YEAR, MULTYEAR, SAMPLE, SERIAL, CBSERIAL, STATEFIP, PUMA, OWNERSHP, RENTGRS, HHINCOME, 
                          ROOMS, BEDROOMS, SEX, AGE, RACE, EDUC, EMPSTAT, WKSWORK1, OCC
2. Place file in local directory
3. Update `path` variable in script:
   path = "YOUR_LOCAL_FILE_PATH.csv"
4. Install required packages:
   pandas, numpy, sklearn

RESULT:
- Script runs successfully end-to-end after path update
- No code modifications required beyond file location update
- Output files generated in /results folder

USER TEST STATUS: PASSED (with dataset dependency noted)
"""