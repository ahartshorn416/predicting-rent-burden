import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# =========================
# 1. LOAD DATA
# =========================
path = r"C:\\Users\\alica\\Downloads\\usa_00003.csv"
df = pd.read_csv(path)

print(f"Data loaded: {df.shape}")

# =========================
# 2. TARGET ENGINEERING
# =========================
df["rent_burdened"] = np.where(
    (df["RENTGRS"] > 0) &
    (df["HHINCOME"] > 0) &
    ((df["RENTGRS"] / df["HHINCOME"]) > 0.3),
    1, 0
)

print("\nrent_burdened value counts:")
print(df["rent_burdened"].value_counts())

# =========================
# 3. FEATURE ENGINEERING
# =========================

# log transform income safely
df["log_income"] = np.log1p(df["HHINCOME"].clip(lower=0))
df["log_rent"] = np.log1p(df["RENTGRS"].clip(lower=0))

# ratio feature (important predictor)
df["rent_income_ratio"] = df["RENTGRS"] / (df["HHINCOME"] + 1)

# unstable employment (your definition using EMPSTAT + WKSWORK1)
df["UNSTABLE_EMPLOYMENT"] = np.where(
    (df["WKSWORK1"] < 35) | (df["EMPSTAT"].isin([2, 3])),
    1, 0
)

print("\nUNSTABLE_EMPLOYMENT value counts:")
print(df["UNSTABLE_EMPLOYMENT"].value_counts())

# =========================
# 4. SELECT MODEL FEATURES
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

df_model = df[features + [target]].dropna()

print("\nFinal modeling dataset:", df_model.shape)

# =========================
# 5. TRAIN / TEST SPLIT
# =========================
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================
# 6. UNSUPERVISED FEATURE ENGINEERING (PCA)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=0.90)  # keep 90% variance
X_pca = pca.fit_transform(X_scaled)

print("\nPCA components retained:", pca.n_components_)

# =========================
# 7. SAFE OUTPUT EXPORT
# =========================
output_dir = r"C:\\Users\\alica\\OneDrive\\Documents\\predicting-rent-burden\\results"

# FIX: create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

df_model.to_csv(os.path.join(output_dir, "model_data.csv"), index=False)

# also save train/test
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

print("\nProcessed data saved successfully.")