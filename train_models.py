import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🚀 Starting Machine Learning Training Pipeline...")

# 1. Load Data
data_path = Path("European_Bank.csv")
if not data_path.exists():
    raise FileNotFoundError("❌ Cannot find European_Bank.csv! Make sure it is in this folder.")

df = pd.read_csv(data_path)

# 2. Feature Engineering (Matching app.py logic)
# Convert categorical strings to dummy variables
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

# 4. Train Model
print("🏋️‍♂️ Training Gradient Boosting Classifier (scikit-learn 1.8.0)...")
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate and Save Assets
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Test Accuracy: {acc*100:.2f}%")

joblib.dump(model, models_dir / "gradient_boosting.pkl")
print("💾 All assets successfully generated and saved to the 'models/' directory!")