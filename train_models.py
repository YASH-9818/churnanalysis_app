import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🚀 Starting Machine Learning Training Pipeline for Dual Models...")

# 1. Load Data
data_path = Path("European_Bank.csv")
if not data_path.exists():
    raise FileNotFoundError("❌ Cannot find European_Bank.csv! Make sure it is in this folder.")

df = pd.read_csv(data_path)

# 2. Feature Engineering (Matching app.py logic)
df_encoded = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)

# Create interaction features
df_encoded["BalanceToSalary"]       = df_encoded["Balance"] / (df_encoded["EstimatedSalary"] + 1)
df_encoded["ProductDensity"]        = df_encoded["NumOfProducts"] / (df_encoded["Tenure"] + 1)
df_encoded["EngagementProduct"]     = df_encoded["IsActiveMember"] * df_encoded["NumOfProducts"]
df_encoded["AgeTenureInteraction"]  = df_encoded["Age"] * df_encoded["Tenure"]
df_encoded["HighBalance"]           = (df_encoded["Balance"] > 100000).astype(int)
df_encoded["SeniorCustomer"]        = (df_encoded["Age"] >= 45).astype(int)

# Drop unnecessary identifier columns safely
drop_cols = ["RowNumber", "CustomerId", "Surname", "Exited"]
X_cols = [col for col in df_encoded.columns if col not in drop_cols]

X = df_encoded[X_cols].astype(np.float32)
y = df_encoded["Exited"].astype(int)

# Save feature names so app.py reads them in the exact same order
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
with open(models_dir / "feature_cols.json", "w") as f:
    json.dump(list(X_cols), f)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Model 1: Gradient Boosting
print("🏋️‍♂️ Training Model 1: Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
print(f"✅ GB Test Accuracy: {gb_acc*100:.2f}%")
joblib.dump(gb_model, models_dir / "gradient_boosting.pkl")

# 5. Train Model 2: Random Forest
print("🏋️‍♂️ Training Model 2: Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"✅ RF Test Accuracy: {rf_acc*100:.2f}%")
joblib.dump(rf_model, models_dir / "random_forest.pkl")

print("💾 All model files (.pkl) successfully saved to the 'models/' directory!")