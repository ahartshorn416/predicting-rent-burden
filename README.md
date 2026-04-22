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
- **Size:** ~16 million observations  

---

## 🧾 Selected Variables
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
- **Size:** ~16 million observations  

---

## 🧾 Selected Variables
YEAR, MULTYEAR, SAMPLE, SERIAL, CBSERIAL,
STATEFIP, PUMA, OWNERSHP, RENTGRS, HHINCOME,
ROOMS, BEDROOMS, SEX, AGE, RACE, EDUC,
EMPSTAT, WKSWORK1, OCC

---

## 🧠 Target Variable
rent_burdened = 1 if (RENTGRS / HHINCOME) > 0.3 else 0


---

## 🛠️ Feature Engineering
- Created **rent-to-income ratio**
- Generated **UNSTABLE_EMPLOYMENT** indicator:
  - Flagged households with low weeks worked or unstable employment status
- Applied **log transformations** to reduce skew (e.g., income)
- Selected key predictors (~10–15 variables) for modeling
- Handled extreme values and invalid entries (e.g., negative income)

---

## ⚙️ Preprocessing
- Removed invalid observations (e.g., zero/negative income for ratio calculation)
- Handled missing values using filtering and transformation
- Split data into **train (80%) / test (20%)**
- Ensured no data leakage (transformations applied after split when appropriate)

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

## 🤖 Models (Planned / In Progress)
- Logistic Regression  
- Random Forest  
- Gradient Boosting  

---

## 📊 Evaluation Metrics
- Precision-Recall AUC (primary metric due to class imbalance)  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

**Baseline PR-AUC:** ~0.0046  
(Target is significantly above baseline)

---

## ⚖️ Fairness Analysis
- Demographic Parity  
- Equalized Odds  
- Group-specific Recall  

Evaluated across:
- Race  
- Sex  
- Geographic regions  

---

## 📌 Key Findings (EDA Stage)
- Rent burden is **rare (~0.46%)**, indicating strong class imbalance  
- Income and rent variables are **highly skewed**, requiring transformation  
- A large portion of households show **unstable employment (~58%)**  
- Zero rent values suggest inclusion of non-renters, requiring filtering  

---

## ⚠️ Limitations
- Cross-sectional data (no time trends)  
- Self-reported income and rent  
- Severe class imbalance  
- Observational data limits causal conclusions  
- Potential bias in feature construction  

---

## 👥 Stakeholder
- U.S. Department of Housing and Urban Development (HUD)  

---

## 🚀 Next Steps
- Train and evaluate predictive models  
- Tune models for imbalanced classification  
- Conduct fairness analysis  
- Interpret results and generate policy insights  

---

## 📚 Sources
- U.S. Census Bureau — American Community Survey (ACS)  
- IPUMS USA — ACS Microdata  
- HUD — Housing Reports  
- Joint Center for Housing Studies (Harvard University)  

---
