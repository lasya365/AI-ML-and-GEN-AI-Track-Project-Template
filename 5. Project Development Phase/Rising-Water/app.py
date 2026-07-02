"""
Rising Waters - Flood Prediction Flask Web Application
Loads the best-performing trained model and serves a prediction UI
for meteorologists / disaster management teams.
"""
import json
import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL = joblib.load("model/best_model.pkl")
SCALER = joblib.load("model/scaler.pkl")
with open("model/results.json") as f:
    META = json.load(f)

FEATURES = META["features"]


def build_feature_vector(form):
    """Order form inputs to match training feature order."""
    return [float(form[f]) for f in FEATURES]


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES, meta=META)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form
        vector = build_feature_vector(data)
        scaled = SCALER.transform([vector])
        pred = int(MODEL.predict(scaled)[0])
        proba = MODEL.predict_proba(scaled)[0]
        flood_prob = float(proba[1]) * 100

        if flood_prob >= 75:
            risk_level = "Severe"
        elif flood_prob >= 50:
            risk_level = "High"
        elif flood_prob >= 25:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        result = {
            "prediction": pred,
            "flood_probability": round(flood_prob, 2),
            "risk_level": risk_level,
            "model_used": META["best_model"],
        }
        if request.is_json:
            return jsonify(result)
        return render_template("index.html", features=FEATURES, meta=META, result=result, form=data)
    except Exception as e:
        error = str(e)
        if request.is_json:
            return jsonify({"error": error}), 400
        return render_template("index.html", features=FEATURES, meta=META, error=error)


@app.route("/about")
def about():
    return render_template("about.html", meta=META)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
