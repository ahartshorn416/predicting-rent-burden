"""
eda.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

This script is my exploratory data analysis on the 2024 ACS 5-Year PUMS data.
The main things I wanted to figure out before modeling were:
  1. How skewed are income and rent? (Spoiler: very -- both needed log transforms)
  2. How bad is the class imbalance in the target variable?
  3. Does UNSTABLE_EMPLOYMENT actually capture what I think it does?
  4. Are there any data quality issues I need to deal with before modeling?
     (The IPUMS sentinel value for missing income was the big one)

Roles:
- Settings: AH
- Key Columns: AH
- Missing Exploration: AH
- Restricting to Renter-only Households: AH
- Create Rent_burdened: AH
- Create Unstable_employment: AH
- Positive class baseline: AH
- Select Variables: AH
- Tables: AH
- Stratified Summaries: AH
- Visualizations: AH
- Testing & Validation: AH
"""

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


# preprocessing.py lives at the repo root, one level up from this script's
# subfolder. This tells Python where to find it so the import works
# regardless of which directory the script is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared preprocessing module: see preprocessing.py for the cleaning
# decisions and why I made them. I'm importing the constants here too
# so I can use IPUMS_INCOME_SENTINEL in the missingness check below
# without hardcoding 9999999 directly in this script.
from preprocessing import (
    load_and_clean,
    build_features,
    get_target,
    IPUMS_INCOME_SENTINEL,
)

# -----------------------------
# Settings
# -----------------------------
sns.set(style="whitegrid")

# I keep results in a separate folder so I'm not accidentally committing
# large PNGs and CSVs to the repo every time I run the script.
results_folder = r"C:\Users\alica\OneDrive\Documents\predicting-rent-burden\results"

# -----------------------------
# Load Raw Data (before cleaning)
# -----------------------------
# I need to look at the raw data before cleaning it so I can document
# the actual missingness and sentinel counts in the EDA report.
# EDA also needs a few extra columns (MULTYEAR for the year trend plot,
# OWNERSHP for ownership type) that the modeling scripts don't need.
data_path   = r"C:\Users\alica\Downloads\usa_00003.csv"
cols_needed = [
    'MULTYEAR', 'HHINCOME', 'RENTGRS', 'WKSWORK1', 'EMPSTAT',
    'AGE', 'EDUC', 'SEX', 'RACE', 'OWNERSHP'
]

df_raw = pd.read_csv(data_path, usecols=cols_needed)
print(f"Raw data loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(df_raw.head())
print(df_raw.tail())

# -----------------------------
# Key Columns
# -----------------------------
# Just naming these so I'm not typing the strings everywhere and
# so it's easy to update if the column names ever change.
income_col  = 'HHINCOME'
rent_col    = 'RENTGRS'
weeks_col   = 'WKSWORK1'
empstat_col = 'EMPSTAT'
target      = get_target()

# -----------------------------
# Missingness Exploration
# -----------------------------
# I wanted to check missingness on the raw data before doing anything
# to it. The ACS doesn't have much true missingness. Most of what
# shows up as "missing" is actually IPUMS sentinel codes that look
# like real values until you know to look for them.
print("\n--- MISSINGNESS ANALYSIS ---")
missing_summary = pd.DataFrame({
    'Missing Count': df_raw.isnull().sum(),
    'Missing %':     (df_raw.isnull().mean() * 100).round(2)
})
missing_summary = missing_summary[
    missing_summary['Missing Count'] > 0
].sort_values('Missing %', ascending=False)
print(missing_summary)
missing_summary.to_csv(os.path.join(results_folder, 'missing_summary.csv'))

# Check how many rows have the IPUMS sentinel value and how many are
# non-renters. I need both counts for the EDA report.
print(f"\nHHINCOME = {IPUMS_INCOME_SENTINEL} (IPUMS missing code) count: "
      f"{(df_raw[income_col] == IPUMS_INCOME_SENTINEL).sum()}")
print(f"RENTGRS = 0 (non-renter or N/A) count: "
      f"{(df_raw[rent_col] == 0).sum()}")

# Show what missingness looks like after replacing the sentinel, so I
# can see how many true NaNs are left before imputation kicks in.
df_sentinel_replaced = df_raw.copy()
df_sentinel_replaced[income_col] = df_sentinel_replaced[income_col].replace(
    IPUMS_INCOME_SENTINEL, np.nan
)
key_vars = [income_col, rent_col, weeks_col, empstat_col]
print("\nMissingness in key variables after sentinel replacement:")
for var in key_vars:
    if var in df_sentinel_replaced.columns:
        n_missing = df_sentinel_replaced[var].isnull().sum()
        pct = n_missing / len(df_sentinel_replaced) * 100
        print(f"  {var}: {n_missing} ({pct:.2f}%)")

# -----------------------------
# Clean Data
# -----------------------------
# load_and_clean() handles the sentinel replacement, median imputation,
# and renter filter. I'm calling it here instead of repeating those
# steps. See preprocessing.py for why I made each decision.
df = load_and_clean(data_path, cols_needed=cols_needed)

# -----------------------------
# Feature Engineering
# -----------------------------
# build_features() creates the target variable and UNSTABLE_EMPLOYMENT
# using the same definitions as the modeling scripts. I used the shared
# function here specifically so EDA and modeling are always consistent.
# Before I centralized this, eda.py was using EMPSTAT != 1 while the
# other scripts used EMPSTAT.isin([2, 3]), which gave slightly different
# results without any error being raised.
df = build_features(df)

print("\nrent_burdened value counts:")
print(df[target].value_counts())
print(f"Positive class rate (rent burdened): {df[target].mean():.4f}")

print("\nUNSTABLE_EMPLOYMENT value counts:")
print(df['UNSTABLE_EMPLOYMENT'].value_counts())

# -----------------------------
# Positive Class Baseline
# -----------------------------
# The naive baseline PR-AUC is just the positive class rate. It's what
# a model that always predicts the majority class would get. I'm noting
# it here so I have a clear floor to compare against when I start modeling.
baseline = df[target].mean()
print(f"\nPositive class baseline (renter-only dataset): {baseline:.4f}")

# -----------------------------
# Select Variables
# -----------------------------
# I split variables into categorical and continuous so I can apply the
# right summary statistics to each. I also check whether each column
# actually exists before appending it, since the optional columns
# (like AGEP) aren't always in the extract.
categorical_vars = ['UNSTABLE_EMPLOYMENT', 'EMPSTAT']
continuous_vars  = [income_col, rent_col]

for col in ['AGE', 'AGEP', 'EDUC', 'SEX', 'RACE', weeks_col]:
    if col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            continuous_vars.append(col)
        else:
            categorical_vars.append(col)

# -----------------------------
# Tables
# -----------------------------
# Frequency tables for the categorical variables. Mainly I wanted to
# see the class breakdown for UNSTABLE_EMPLOYMENT and EMPSTAT to confirm
# they were behaving as expected (~54% unstable made sense for renters).
print("\n--- CATEGORICAL SUMMARIES ---")
for var in categorical_vars:
    if var in df.columns:
        freq    = df[var].value_counts(dropna=False)
        prop    = df[var].value_counts(normalize=True, dropna=False)
        summary = pd.DataFrame({'Count': freq, 'Proportion': prop})
        print(f"\n{var}")
        print(summary.head())

# Descriptive stats and skewness for continuous variables. I specifically
# wanted to see the skewness numbers for income and rent. They came back
# at 4.31 and 2.03, which confirmed I needed log transforms before modeling.
print("\n--- CONTINUOUS SUMMARIES ---")
for var in continuous_vars:
    if var in df.columns:
        print(f"\n{var}")
        print(df[var].describe())
        print(f"Skew: {df[var].skew():.2f}, Kurtosis: {df[var].kurtosis():.2f}")

# -----------------------------
# Stratified Summaries
# -----------------------------
# Breaking down income and rent by rent burden status was a basic sanity
# check. If the target variable is capturing real signal, burdened and
# non-burdened households should look noticeably different on these variables.
for var in continuous_vars:
    if var in df.columns:
        summary = df.groupby(target, observed=True)[var].describe()
        summary.to_csv(os.path.join(results_folder, f"{var}_by_target.csv"))

# -----------------------------
# Visualizations
# -----------------------------
# The full renter dataset is ~3.6M rows, which is too slow for seaborn
# pairplots and histograms. I sampled 50K rows which is enough to get
# stable distributions without waiting 10+ minutes for a plot to render.
df_sample = df.sample(n=50000, random_state=42)

# 1. Income histogram: I expected this to be right-skewed and it was.
#    The long tail is mostly high-income renters in expensive cities.
plt.figure()
sns.histplot(df_sample[income_col], bins=50, kde=True)
plt.title("Household Income Distribution (Renters Only)")
plt.savefig(os.path.join(results_folder, "income_hist.png"))
plt.close()

# 2. Rent histogram: also right-skewed for the same reason. High rents
#    in NYC/SF/Boston pull the tail out even though most people pay
#    something in the middle. This confirmed I needed log_rent too.
plt.figure()
sns.histplot(df_sample[rent_col], bins=50, kde=True)
plt.title("Rent Distribution (Renters Only)")
plt.savefig(os.path.join(results_folder, "rent_hist.png"))
plt.close()

# 3. Employment bar chart: I wanted to see whether UNSTABLE_EMPLOYMENT
#    was actually capturing a meaningful portion of the renter population
#    or if it was going to be a near-zero feature. ~54% came back unstable,
#    which made sense given how many renters work part-time or seasonally.
plt.figure()
sns.countplot(x='UNSTABLE_EMPLOYMENT', data=df_sample)
plt.title("Unstable Employment (Renters Only)")
plt.savefig(os.path.join(results_folder, "unstable_employment.png"))
plt.close()

# 4. Income boxplot by rent burden: just a visual sanity check that
#    burdened and non-burdened households actually look different on income.
plt.figure()
sns.boxplot(x=target, y=income_col, data=df_sample)
plt.title("Income by Rent Burden")
plt.savefig(os.path.join(results_folder, "income_boxplot.png"))
plt.close()

# 5. Correlation heatmap: I wanted to check for multicollinearity before
#    modeling. The highest pairwise correlation was 0.50 (WKSWORK1 vs. EDUC),
#    which isn't a problem for logistic regression.
cont_vars_exist = [v for v in continuous_vars if v in df.columns]
corr = df_sample[cont_vars_exist].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig(os.path.join(results_folder, "correlation_matrix.png"))
plt.close()

# 6. Pairplot: I capped this at 5 variables so the plot stays readable.
#    Coloring by target lets me see at a glance which features separate
#    the two classes.
pair_vars       = cont_vars_exist[:5] + [target]
pairplot_sample = df_sample[pair_vars].copy()
sns.pairplot(pairplot_sample, diag_kind='kde', hue=target)
plt.savefig(os.path.join(results_folder, "pairplot.png"))
plt.close()

# 7. Rent burden rate by year: the 5-year ACS pools multiple survey years,
#    so I wanted to check whether burden rates were stable across years or
#    trending in a way that might confound the analysis. If the rates were
#    shifting a lot, treating the pooled data as a single cross-section
#    would be harder to justify.
if 'MULTYEAR' in df.columns:
    rent_by_year          = df.groupby('MULTYEAR')[target].mean().reset_index()
    rent_by_year.columns  = ['MULTYEAR', 'rent_burden_rate']

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=rent_by_year, x='MULTYEAR', y='rent_burden_rate', marker='o'
    )
    plt.title("Rent Burden Rate per Year (Renters Only)")
    plt.xlabel("Year")
    plt.ylabel("Proportion Rent Burdened")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "rent_burden_per_year.png"))
    plt.close()
    print("\nRent burden by year:")
    print(rent_by_year)
else:
    print("Warning: 'MULTYEAR' column not found -- skipping year plot.")

print("\nEDA complete -- all outputs saved.")

# -----------------------------
# Testing & Validation
# -----------------------------

"""
SELF TEST: AH on Local Machine
------------------------------
- Script runs without errors in PyCharm
- Missingness checked on raw data before any cleaning
- IPUMS sentinel (9999999) confirmed replaced -- checked via print statement
- Row count drops from ~16.1M to ~3.6M after renter filter, as expected
- Positive class rate: 3.65% -- consistent with what I expected from HUD data
- UNSTABLE_EMPLOYMENT: ~54% flagged as unstable, which makes sense for renters
- All summary stats computed and saved correctly
- All 7 visualizations generated without errors
- Used 50K row sample for plots -- generation time under 2 minutes

USER TEST: Secondary Device
---------------------------
- Script is designed to be portable
- Raw IPUMS data not included in repo due to file size
- Download from: https://usa.ipums.org/usa/

REQUIRED USER STEPS:
1. Download 2024 ACS 5-Year dataset from IPUMS (CSV format)
   Variables needed: YEAR, MULTYEAR, SAMPLE, SERIAL, CBSERIAL, STATEFIP,
                     PUMA, OWNERSHP, RENTGRS, HHINCOME, ROOMS, BEDROOMS,
                     SEX, AGE, RACE, EDUC, EMPSTAT, WKSWORK1, OCC
2. Save file locally
3. Update data_path in this script
4. Update results_folder to your output directory
5. Install: pandas, numpy, matplotlib, seaborn

USER TEST STATUS: PASSED (with dataset dependency noted)
"""