"""
Trains and compares four classifiers for flood prediction using the
real historical flood dataset (Temp, Humidity, Cloud Cover, rainfall
periods, flood label). Dataset is small (115 rows) and imbalanced
(~14% flood-positive), so models use class_weight/scale_pos_weight
and stratified splitting.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

FEATURES = ["Temp", "Humidity", "Cloud Cover", "ANNUAL", "Jan-Feb",
            "Mar-May", "Jun-Sep", "Oct-Dec", "avgjune", "sub"]
TARGET = "flood"

df = pd.read_csv("../data/flood_data.csv")
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# class imbalance ratio for XGBoost
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos if pos > 0 else 1

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42
    ),
}

results = {}
trained = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()
    results[name] = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": cm,
    }
    trained[name] = model
    print(f"{name:15s} | Accuracy: {acc*100:.2f}%  Precision: {prec*100:.2f}%  "
          f"Recall: {rec*100:.2f}%  F1: {f1*100:.2f}%")

# Pick best by F1 (not raw accuracy) since the dataset is imbalanced —
# a model that always predicts "no flood" would score ~86% accuracy
# while being useless. F1 balances catching real floods vs false alarms.
best_name = max(results, key=lambda n: results[n]["f1_score"])
best_model = trained[best_name]
print(f"\nBest model (by F1): {best_name} ({results[best_name]['f1_score']}% F1, "
      f"{results[best_name]['accuracy']}% accuracy)")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

with open("results.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "features": FEATURES,
        "all_results": results
    }, f, indent=2)

if hasattr(best_model, "feature_importances_"):
    importances = dict(zip(FEATURES, best_model.feature_importances_.round(4).tolist()))
    print("\nFeature importances:", importances)