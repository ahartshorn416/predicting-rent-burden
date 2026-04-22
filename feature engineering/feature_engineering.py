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

- Target Engineering: AH
- Feature Engineering: AH
- Select Model Features: AH
- Train Test Split: AH
- Unsupervised Feature Engineering (PCA): AH
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

os.makedirs(output_dir, exist_ok=True)

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
        (df["RENTGRS"] > 0) &
        (df["HHINCOME"] > 0) &
        ((df["RENTGRS"] / df["HHINCOME"]) > 0.3),
        1, 0
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


#  -----------------------------
# Load Data
#  -----------------------------
df = pd.read_csv(path)
print(f"Data loaded: {df.shape}")

# -----------------------------
# Feature Engineering
# -----------------------------
df = build_rent_burden_features(df)

print("\nTarget distribution:")
print(df["rent_burdened"].value_counts())

print("\nEmployment instability distribution:")
print(df["UNSTABLE_EMPLOYMENT"].value_counts())

# =========================
# 3. SELECT FEATURES
# =========================
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
#  Standardize + PCA
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train_scaled)

print("\nPCA components retained:", pca.n_components_)

# -----------------------------
# Export Files
# -----------------------------
df_model.to_csv(os.path.join(output_dir, "model_data.csv"), index=False)
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

print("\nAll processed files saved successfully.")