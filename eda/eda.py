"""
eda.py

Author: Alison Hartshorn
Team: Iota
Project Name: Predicting Rent Burden in U.S. Households Using Machine Learning and
Fairness Analysis

This script performs exploratory data analysis on U.S. household survey data by creating a rent-burden indicator,
summarizing key demographic and economic variables, and generating statistical tables and visualizations that are
saved to an output folder.

Roles:

- Settings: AH
- Key Columns: AH
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

# -----------------------------
# Settings
# -----------------------------
sns.set(style="whitegrid")
results_folder = r"C:\Users\alica\OneDrive\Documents\prediciting_rent_burden_shocks\results"
os.makedirs(results_folder, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
data_path = r"C:\\Users\\alica\\Downloads\\usa_00003.csv"
df = pd.read_csv(data_path)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------------
# Key Columns
# -----------------------------
income_col = 'HHINCOME'
rent_col = 'RENTGRS'
weeks_col = 'WKSWORK1'   # FIXED
empstat_col = 'EMPSTAT'
target = 'rent_burdened'

# -----------------------------
# Create rent_burdened
# -----------------------------
df[target] = np.where(
    (df[income_col] > 0) & (df[rent_col] / df[income_col] > 0.3),
    1,
    0
)

print("\nrent_burdened value counts:")
print(df[target].value_counts())

# -----------------------------
# Create UNSTABLE_EMPLOYMENT
# -----------------------------
if weeks_col in df.columns and empstat_col in df.columns:
    df['UNSTABLE_EMPLOYMENT'] = np.where(
        (df[weeks_col] < 35) | (df[empstat_col] != 1),
        1,
        0
    )
else:
    print("Warning: Missing employment columns, using EMPSTAT only")
    df['UNSTABLE_EMPLOYMENT'] = np.where(df[empstat_col] != 1, 1, 0)

print("\nUNSTABLE_EMPLOYMENT value counts:")
print(df['UNSTABLE_EMPLOYMENT'].value_counts())

# -----------------------------
# Positive class baseline
# -----------------------------
baseline = df[target].mean()
print("Positive class baseline:", baseline)

# -----------------------------
# Select variables
# ----------------------------
categorical_vars = ['UNSTABLE_EMPLOYMENT', 'EMPSTAT']
continuous_vars = [income_col, rent_col]

# Add optional variables if they exist
for col in ['AGE', 'AGEP', 'EDUC', 'SEX', 'RACE', weeks_col]:
    if col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            continuous_vars.append(col)
        else:
            categorical_vars.append(col)

# -----------------------------
# Tables
# -----------------------------
print("\n--- CATEGORICAL SUMMARIES ---")
for var in categorical_vars:
    if var in df.columns:
        freq = df[var].value_counts(dropna=False)
        prop = df[var].value_counts(normalize=True, dropna=False)
        summary = pd.DataFrame({'Count': freq, 'Proportion': prop})
        print(f"\n{var}")
        print(summary.head())

print("\n--- CONTINUOUS SUMMARIES ---")
for var in continuous_vars:
    if var in df.columns:
        print(f"\n{var}")
        print(df[var].describe())
        print(f"Skew: {df[var].skew():.2f}, Kurtosis: {df[var].kurtosis():.2f}")

# -----------------------------
# Stratified summaries
# -----------------------------
for var in continuous_vars:
    if var in df.columns:
        summary = df.groupby(target, observed=True)[var].describe()
        summary.to_csv(os.path.join(results_folder, f"{var}_by_target.csv"))

# -----------------------------
# VISUALIZATIONS
# -----------------------------

# Use sample for speed (VERY IMPORTANT with 16M rows)
df_sample = df.sample(n=50000, random_state=42)

# 1. Income histogram
plt.figure()
sns.histplot(df_sample[income_col], bins=50, kde=True)
plt.title("Household Income Distribution")
plt.savefig(os.path.join(results_folder, "income_hist.png"))
plt.close()

# 2. Rent histogram
plt.figure()
sns.histplot(df_sample[rent_col], bins=50, kde=True)
plt.title("Rent Distribution")
plt.savefig(os.path.join(results_folder, "rent_hist.png"))
plt.close()

# 3. Employment bar chart
plt.figure()
sns.countplot(x='UNSTABLE_EMPLOYMENT', data=df_sample)
plt.title("Unstable Employment")
plt.savefig(os.path.join(results_folder, "unstable_employment.png"))
plt.close()

# 4. Boxplot income vs rent burden
plt.figure()
sns.boxplot(x=target, y=income_col, data=df_sample)
plt.title("Income by Rent Burden")
plt.savefig(os.path.join(results_folder, "income_boxplot.png"))
plt.close()

# 5. Correlation heatmap
cont_vars_exist = [v for v in continuous_vars if v in df.columns]
corr = df_sample[cont_vars_exist].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig(os.path.join(results_folder, "correlation_matrix.png"))
plt.close()

# 6. Pairplot
pair_vars = cont_vars_exist[:5] + [target]
pairplot_sample = df_sample[pair_vars].copy()

sns.pairplot(pairplot_sample, diag_kind='kde', hue=target)
plt.savefig(os.path.join(results_folder, "pairplot.png"))
plt.close()

print("\n✅ EDA COMPLETE — all outputs saved.")
print("EDA completed and figures saved.")

# -----------------------------
# Testing & Validation
# -----------------------------

"""
SELF TEST: AH on Local Machine
------------------------------
- Script runs successfully on Pycharm
- No runtime errors during full EDA pipeline execution
- Dataset loaded successfully from local IPUMS CSV file
- Target variable (rent_burdened) created without errors
- UNSTABLE_EMPLOYMENT feature generated successfully
- Summary statistics computed for both categorical and continuous variables
- Visualizations generated and saved to output directory
- Sampling (n=50,000) successfully used for performance optimization

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