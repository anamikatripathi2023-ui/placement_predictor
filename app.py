# app.py
# Flask app that serves the frontend and a /predict API.
# Place model.joblib and scaler.joblib in the same folder before starting.

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load persisted artifacts at startup
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded model from {MODEL_PATH} and scaler from {SCALER_PATH}")
except Exception as e:
    print("ERROR loading model/scaler:", e)
    model = None
    scaler = None

@app.route("/")
def index():
    return render_template("index.html", title="Placement Predictor")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded on server."}), 500

    # Accept JSON or form-encoded requests
    data = request.get_json(silent=True)
    if data is None:
        data = request.form

    try:
        cgpa = float(data.get("cgpa"))
        iq = float(data.get("iq"))
    except Exception:
        return jsonify({"error": "Please provide numeric 'cgpa' and 'iq' in JSON or form data."}), 400

    X = np.array([[cgpa, iq]])
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_scaled)[0, 1])

    return jsonify({"placement": pred, "probability": proba})

if __name__ == "__main__":
    # For development only. Use Gunicorn in production.
    app.run(host="0.0.0.0", port=5000, debug=True)
