# save_model_snippet.py
# Run this (or paste into your notebook) to train and save a model + scaler.
# Adjust CSV path if needed.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load data (update path if run from a different folder)
df = pd.read_csv("placement.csv")  # place placement.csv in same folder, or change path

# If your CSV contains an extra index column like 'Unnamed: 0', drop it:
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Features & target
X = df[["cgpa", "iq"]].values
y = df["placement"].values

# Train-test split (set random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate (quick)
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")

# Save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Saved model.joblib and scaler.joblib")
