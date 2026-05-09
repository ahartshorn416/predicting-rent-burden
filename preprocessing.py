"""
preprocessing.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

I created this module because I noticed I was copying the same ~40 lines of
preprocessing code into every script, and they were starting to drift apart.
The eda.py version used EMPSTAT != 1 for employment instability while the
modeling scripts used EMPSTAT.isin([2, 3]) -- they're close but not the same,
and that kind of inconsistency can cause silent differences in results.

Putting everything in one place means if I change a threshold or fix a bug,
it updates everywhere automatically instead of me having to track down four
copies of the same code.

Functions:
    load_and_clean(path, cols_needed)  -- loads CSV, fixes sentinel, imputes, filters
    build_features(df)                 -- engineers all features and target variable
    get_feature_list()                 -- returns the 6-feature model list
    get_target()                       -- returns the target column name
"""

import pandas as pd
import numpy as np

# -----------------------------
# Constants
# -----------------------------

# IPUMS codes missing HHINCOME as 9999999 -- it's in their codebook.
# I'm storing it as a constant so it's obvious what the number means
# and easy to find if I ever need to change it.
IPUMS_INCOME_SENTINEL = 9999999

# HUD's 30% threshold: this is the standard definition of rent burden
# used in housing policy research, so I used it as-is.
RENT_BURDEN_THRESHOLD = 0.30

# I used 35 weeks as the cutoff for employment instability because it's
# a common threshold for part-time/seasonal work in labor economics.
UNSTABLE_WEEKS_THRESHOLD = 35

# EMPSTAT 2 = unemployed, 3 = not in labor force. I excluded code 0
# (N/A, e.g. children) because those aren't really "unstable" workers.
# They're just not expected to be working at all.
UNSTABLE_EMPSTAT_CODES = [2, 3]

# These are the six features I ended up using in all the models after
# I caught the leakage issue. I originally included log_income, log_rent,
# and rent_income_ratio, but an initial run gave me ROC-AUC = 1.0, which
# was a red flag. Those features basically reconstruct the target variable,
# so I removed them and stuck to demographic and employment predictors only.
MODEL_FEATURES = [
    "AGE",
    "EDUC",
    "SEX",
    "RACE",
    "WKSWORK1",
    "UNSTABLE_EMPLOYMENT",
]

TARGET = "rent_burdened"


# -----------------------------
# Functions
# -----------------------------

def get_feature_list():
    """
    Returns the list of model features used across all scripts.

    I'm using a function instead of just importing MODEL_FEATURES directly
    so that each script gets its own copy and can't accidentally modify
    the shared list.

    Returns:
        list[str]: The six demographic and employment feature names.
    """
    return MODEL_FEATURES.copy()


def get_target():
    """
    Returns the target column name ('rent_burdened').

    Returns:
        str: 'rent_burdened'
    """
    return TARGET


def load_and_clean(path, cols_needed=None):
    """
    Loads the ACS PUMS CSV and runs the three cleaning steps I need
    before any feature engineering can happen.

    The order here matters:
        1. Load (with usecols to avoid memory errors on the full 16M row file)
        2. Replace IPUMS sentinel before anything else touches HHINCOME
        3. Median impute the remaining NaNs
        4. Filter to renters only

    I need to fix the sentinel before I filter to renters because the
    target variable gets built next and it uses HHINCOME in the ratio.
    If I leave 9999999 in there, those rows look like $10M incomes and
    get incorrectly labeled as not rent-burdened.

    Args:
        path (str): Path to the IPUMS ACS CSV file.
        cols_needed (list[str], optional): Columns to load. Defaults to the
            minimal set for modeling. EDA scripts pass a broader list since
            they need things like MULTYEAR and OWNERSHP.

    Returns:
        pd.DataFrame: Cleaned renter-only dataset, usually ~3.56M rows.
    """
    if cols_needed is None:
        # Minimal set for modeling: I use usecols here because loading
        # all ~70 ACS columns at once causes memory errors on my machine.
        cols_needed = [
            'HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT',
            'AGE', 'EDUC', 'SEX', 'RACE'
        ]

    df = pd.read_csv(path, usecols=cols_needed)
    print(f"Data loaded: {df.shape}")

    # Step 1: Replace IPUMS sentinel with NaN.
    # This has to happen first. If I impute or filter before catching
    # this, the $10M values contaminate everything downstream.
    df['HHINCOME'] = df['HHINCOME'].replace(IPUMS_INCOME_SENTINEL, np.nan)

    # Step 2: Median impute.
    # I went with median over mean because HHINCOME is really right-skewed
    # (skewness ~3.88 from EDA). The mean gets pulled up by high-income
    # outliers and would overstate what a typical household earns.
    # Only about 0.4% of rows are affected so it doesn't change much,
    # but I wanted to handle it correctly.
    df['HHINCOME'] = df['HHINCOME'].fillna(df['HHINCOME'].median())

    # Step 3: Keep renters only.
    # Non-renters can't be rent-burdened by definition, and including them
    # tanks the positive class rate from 3.65% to ~0.46%, which makes
    # PR-AUC almost meaningless as an evaluation metric.
    n_before = len(df)
    df = df[df['RENTGRS'] > 0].copy()
    print(f"Restricted to renters: {n_before - len(df)} rows removed "
          f"({n_before} to {len(df)})")

    return df


def build_features(df):
    """
    Builds the target variable and all the engineered features.

    I'm creating log_income, log_rent, and rent_income_ratio here so
    EDA scripts can use them for exploration, but I excluded them from
    MODEL_FEATURES because they caused target leakage. When I included
    them in an early run, the model got ROC-AUC = 1.0, which is obviously
    wrong. It was just reconstructing the target from the same income
    and rent numbers used to define it.

    Args:
        df (pd.DataFrame): Cleaned renter-only dataset from load_and_clean().
            Needs: HHINCOME, RENTGRS, WKSWORK1, EMPSTAT.

    Returns:
        pd.DataFrame: Copy of df with the engineered columns added.
    """
    df = df.copy()

    # --- Target variable ---
    # HUD defines rent burden as spending more than 30% of income on rent.
    # For households with zero or negative income, I assign rent_burdened = 1
    # directly,  any rent at all exceeds 30% of $0, and it avoids a
    # division-by-zero error in the ratio.
    df[TARGET] = np.where(
        df["HHINCOME"] <= 0,
        1,
        np.where(
            (df["RENTGRS"] / df["HHINCOME"]) > RENT_BURDEN_THRESHOLD, 1, 0
        )
    )

    # --- Log transforms (EDA only -- not used in models) ---
    # Both income and rent are heavily right-skewed, so I log-transformed
    # them for EDA visualizations. I used log1p instead of log because
    # log(0) is undefined, and clip(lower=0) handles any negatives safely.
    df["log_income"] = np.log1p(df["HHINCOME"].clip(lower=0))
    df["log_rent"]   = np.log1p(df["RENTGRS"].clip(lower=0))

    # --- Rent-to-income ratio (EDA only -- not used in models) ---
    # The +1 in the denominator is just to avoid division-by-zero for
    # any rows where HHINCOME is exactly 0 after imputation.
    df["rent_income_ratio"] = df["RENTGRS"] / (df["HHINCOME"] + 1)

    # --- Employment instability flag ---
    # I flag a household as unstable if the person worked fewer than
    # 35 weeks OR is currently unemployed/not in the labor force.
    # Either condition on its own is enough to put rent payments at risk,
    # so I combined them with OR rather than requiring both.
    df["UNSTABLE_EMPLOYMENT"] = np.where(
        (df["WKSWORK1"] < UNSTABLE_WEEKS_THRESHOLD) |
        (df["EMPSTAT"].isin(UNSTABLE_EMPSTAT_CODES)),
        1, 0
    )

    return df
