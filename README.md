# 🏠 Predicting Rent Burden in U.S. Households Using Machine Learning & Fairness Analysis

## 📌 Overview
This project analyzes whether U.S. households are rent burdened—defined as spending more than 30% of household income on rent—using large-scale survey data and machine learning techniques. In addition to predicting rent burden status, the project evaluates how model performance varies across demographic groups to assess fairness and equity.

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
- **Format:** `.dat` file processed into a structured dataset using Python  
- **Size:** ~16 million observations (restricted to ~3.56M renter-only households for modeling)

---

## 🧾 Selected Variables
YEAR, MULTYEAR, SAMPLE, SERIAL, CBSERIAL,
STATEFIP, PUMA, OWNERSHP, RENTGRS, HHINCOME,
ROOMS, BEDROOMS, SEX, AGE, RACE, EDUC,
EMPSTAT, WKSWORK1, OCC

---

## 🧠 Target Variable
```python
rent_burdened = 1 if (RENTGRS / HHINCOME) > 0.3 else 0
# Households with HHINCOME <= 0 are assigned rent_burdened = 1 by definition
```

---

## 🛠️ Feature Engineering
- Created **rent-to-income ratio** (excluded from model feature set to prevent target leakage)
- Generated **UNSTABLE_EMPLOYMENT** indicator:
  - Flagged households with `WKSWORK1 < 35` or `EMPSTAT ∈ {2, 3}` (unemployed / not in labor force)
- Applied **log transformations** to reduce skew on income and rent variables
- Restricted dataset to **renter-only households** (`RENTGRS > 0`) — non-renters cannot be rent-burdened by definition
- Handled extreme values and invalid entries (e.g., IPUMS sentinel value 9,999,999 for missing income)

---

## ⚙️ Preprocessing
- Replaced IPUMS sentinel income values with NaN; applied median imputation (~0.4% of rows affected)
- Removed non-renter and zero/negative income observations
- Split data into **train (80%) / test (20%)** with stratification on target variable
- `StandardScaler` and `PCA` fit on training data only and applied to test set to prevent data leakage
- Income, rent, and ratio features (`log_income`, `log_rent`, `rent_income_ratio`) excluded from model feature set after initial run produced perfect ROC-AUC = 1.0 (target leakage confirmed)

---

## 📈 Exploratory Data Analysis (EDA)
- Generated summary tables for categorical and continuous variables  
- Assessed skewness and kurtosis for continuous variables  
- Visualizations included:
  - Histograms  
  - Boxplots  
  - Correlation matrix  
  - Pairplots (sampled due to dataset size)  

---

## 🤖 Models

### ✅ Baseline: Logistic Regression (Complete)

The baseline logistic regression model was trained on 6 demographic and employment features: `AGE`, `EDUC`, `SEX`, `RACE`, `WKSWORK1`, and `UNSTABLE_EMPLOYMENT`. Income and rent variables were excluded to prevent target leakage. `class_weight='balanced'` was used to address the 3.65% positive class rate without resampling.

#### Evaluation Metrics (Test Set, n = 712,928)

| Metric | Value | Notes |
|--------|-------|-------|
| ROC-AUC | 0.7730 | Solid discriminative ability for a baseline |
| PR-AUC | 0.0967 | ~2.6x above naive baseline (0.0365) |
| Naive Baseline PR-AUC | 0.0365 | Positive class rate in training set |
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
| 1 | 0.7728 | 0.7714 | 0.9999 | 0.9999 |
| 2 | 0.7726 | 0.7724 | 0.9999 | 0.9999 |
| 3 | 0.7727 | 0.7721 | 0.9999 | 0.9999 |
| 4 | 0.7719 | 0.7752 | 0.9999 | 0.9999 |
| 5 | 0.7728 | 0.7715 | 0.9999 | 0.9999 |
| **Mean (Val)** | — | **0.7725 ± 0.0016** | — | **0.0969 ± 0.0008** |

Training and validation scores are nearly identical across all folds, indicating no evidence of overfitting.

**Key takeaway:** The baseline achieves a PR-AUC ~2.6x above the naive baseline with 92% recall on the minority class. Precision (0.07) is the primary weakness and is expected to improve with tree-based models. The positive coefficient on education is unexpected and will be investigated further.

### 🔜 Upcoming Models
- Random Forest  
- Gradient Boosting  

---

## 📊 Evaluation Metrics
- **Precision-Recall AUC** (primary metric due to class imbalance)  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

**Positive Class Rate (renter-only dataset):** 3.65%  
**Naive Baseline PR-AUC:** 0.0365

---

## ⚖️ Fairness Analysis *(Planned)*
- Demographic Parity  
- Equalized Odds  
- Group-specific Recall  

Evaluated across:
- Race  
- Sex  
- Geographic regions  

---

## 📌 Key Findings

### EDA Stage
- Rent burden is rare (~3.65% among renters), indicating strong class imbalance  
- Income and rent variables are highly skewed, requiring log transformation  
- ~54% of renter households show unstable employment  
- Non-renter filtering reduced dataset from 16.1M to 3.6M rows  

### Baseline Model Stage
- Logistic regression achieves ROC-AUC of 0.773 and PR-AUC of 0.097 using only demographic and employment features  
- Weeks worked (`WKSWORK1`) is the strongest predictor of rent burden (coefficient = -1.288)  
- Unstable employment is the strongest positive predictor (coefficient = +0.357), consistent with the project hypothesis  
- High recall (0.92) makes this baseline well-suited for policy screening applications where missing at-risk households is costly  
- No overfitting detected (mean Val ROC-AUC = 0.773, SD = 0.002 across 5 folds)  

---

## ⚠️ Limitations
- Cross-sectional data (no time trends)  
- Self-reported income and rent  
- Severe class imbalance  
- Observational data limits causal conclusions  
- RACE and EDUC treated as ordinal integers in the baseline model; one-hot encoding of RACE will be considered in the fairness analysis phase  
- Median imputation for HHINCOME may modestly compress income variance near the median (~65,000 rows affected)

---

## 👥 Stakeholder
- U.S. Department of Housing and Urban Development (HUD)  

---

## 🚀 Next Steps
- Train Random Forest and Gradient Boosting classifiers with hyperparameter tuning via `RandomizedSearchCV`  
- Investigate positive coefficient on `EDUC` — potential confound with urban residence  
- Re-introduce income/rent features carefully to measure additional predictive lift above the demographic-only baseline  
- Conduct fairness analysis across race and sex subgroups using demographic parity and equalized odds  

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
git clone <your-repo-url>
cd <your-project-folder>
```

### 📊 2. Download the dataset (REQUIRED)

This project uses IPUMS ACS microdata, which is not included due to file size.

Steps:
1. Go to https://usa.ipums.org/usa/
2. Create a free account
3. Select the 2024 ACS 5-Year dataset and the selected variables listed above
4. Download as CSV format
5. Save it to your local machine

### ⚙️ 3. Update file paths in scripts

Open the Python files and update the dataset path:
```python
path = r"C:\Users\YOUR_USERNAME\Downloads\your_file.csv"
output_dir = r"C:\Users\YOUR_USERNAME\...\results"
```

### 📦 4. Install required packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### ▶️ 5. Run the scripts in order
```bash
python eda.py
python feature_engineering.py
python baseline_model.py
```

### 📂 6. Output files

All outputs will be saved in `/results`, including:
- Cleaned datasets  
- Summary tables  
- Visualizations (ROC curve, PR curve, coefficient plot)  
- Model-ready datasets  
- `baseline_metrics.csv` — evaluation metrics  
- `baseline_coefficients.csv` — logistic regression coefficients  
- `baseline_cv_results.csv` — cross-validation fold results
