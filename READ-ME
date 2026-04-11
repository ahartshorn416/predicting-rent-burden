# 🏠 Predicting Rent Burden in U.S. Households Using Machine Learning and Fairness Analysis

## 📌 Overview  
This project uses machine learning to predict whether U.S. households are rent burdened, defined as spending more than 30% of income on housing. In addition to building predictive models, the project also examines whether model performance differs across demographic and geographic groups to assess fairness and equity.

The broader goal is to better understand which households are most at risk and provide insights that could help policymakers more effectively target housing assistance and improve affordability outcomes.

---

## ❓ Research Question  
Can machine learning models predict which U.S. households are rent burdened, and does model performance vary across demographic and geographic groups in ways that could inform housing policy decisions?

---

## 🎯 Objectives  
- Predict rent burden status using machine learning models  
- Identify key drivers of housing affordability  
- Evaluate model performance on imbalanced data using appropriate metrics  
- Assess fairness across demographic groups  
- Generate insights relevant to housing policy and resource allocation  

---

## 📊 Data Source  
The dataset comes from:

- 2020–2024 American Community Survey (ACS) 5-Year Public Use Microdata Sample (PUMS)  
- IPUMS USA (https://ipums.org)  
- Downloaded as a CSV file  

The dataset contains over 16 million observations and includes demographic, economic, and housing-related variables.

---

## 🧾 Selected Variables  
A subset of key variables (~20–25 predictors) was selected for analysis, including:

- **Housing & Income:** RENTGRS, HHINCOME, ROOMS, BEDROOMS, UNITSSTR  
- **Demographics:** AGE, SEX, RACE, HISPAN, EDUC  
- **Employment:** EMPSTAT, WKSWORK1, OCC  
- **Geography:** STATEFIP, PUMA  

Identifier variables (e.g., SERIAL, SAMPLE) are excluded from modeling.

---

## 🧠 Target Variable  
A binary outcome is constructed:

```python
rent_burdened = 1 if (RENTGRS / HHINCOME) > 0.3 else 0
```

This follows the standard definition used in housing research.

---

## 🔍 Key Insights from EDA  
- Rent burden is **rare (~0.46%)**, creating a highly imbalanced classification problem  
- Household income and rent are **heavily right-skewed** with extreme outliers  
- A large portion of observations have **zero rent**, indicating non-renters  
- Employment instability is common (~58%), suggesting it may be an important predictor  
- Most predictors show **low multicollinearity**, meaning they contribute distinct information  

---

## ⚙️ Methodology  

### Data Processing
- Filter to renter households (RENTGRS > 0)  
- Handle missing values (median imputation)  
- Apply log transformations to skewed variables  
- Create derived features (e.g., UNSTABLE_EMPLOYMENT)  
- Encode categorical variables  

### Models
- Logistic Regression (baseline)  
- Random Forest  
- Gradient Boosting (e.g., XGBoost)  

### Evaluation Metrics
Due to extreme class imbalance:

- **PR-AUC (Primary metric)**  
- Precision & Recall  
- F1-score  
- ROC-AUC (secondary metric)  

Baseline PR-AUC ≈ **0.0046**, so models should significantly exceed this.

---

## ⚖️ Fairness Analysis  
The project evaluates model fairness using:

- Demographic Parity  
- Equalized Odds  
- Group-specific Recall  

This helps identify whether model performance differs across groups such as race, ethnicity, or geography.

---

## 📈 Expected Results  
- Moderate predictive performance (**PR-AUC ~0.02–0.05**)  
- Income and rent as the strongest predictors  
- Employment instability contributing additional signal  
- Variation in performance across demographic groups  

---

## ⚠️ Limitations  
- Cross-sectional data (no time dynamics)  
- Self-reported income and rent  
- Extreme class imbalance  
- Presence of non-renters in raw data  
- Observational data limits causal conclusions  

---

## 👥 Stakeholder  
- U.S. Department of Housing and Urban Development (HUD)

---

## 📚 Sources  
- United States Census Bureau. American Community Survey (ACS)  
- IPUMS USA  
- U.S. Department of Housing and Urban Development (HUD)  
- Joint Center for Housing Studies of Harvard University  

---

## 🚀 Next Steps  
- Finalize preprocessing (filter renters, transformations)  
- Train and compare models  
- Evaluate using PR-AUC  
- Conduct fairness analysis  
- Generate policy-relevant insights  
