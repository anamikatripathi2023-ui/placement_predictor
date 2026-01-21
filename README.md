# Placement Predictor - Minimal Web Integration for ML Model

This repository contains a small example showing how to turn your ML model (trained in the notebook) into a website named "Placement Predictor".

Quick overview:
1. Train & save model and scaler (use `save_model_snippet.py`).
2. Run a Flask server (`app.py`) which loads `model.joblib` and `scaler.joblib`.
3. Open the frontend at `http://localhost:5000/` and input CGPA and IQ to get predictions.

Prerequisites
- Python 3.8+
- pip

Install
1. Create & activate a venv (recommended)
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install dependencies
   pip install -r requirements.txt

Train & save model
- Put `placement.csv` (your dataset) in the project folder.
- Run:
   python save_model_snippet.py
- This will create `model.joblib` and `scaler.joblib`.

Run the app (development)
- Start Flask app:
   python app.py
- Open your browser to http://localhost:5000/

Docker (optional)
- Build:
   docker build -t placement-predictor:latest .
- Run:
   docker run -p 5000:5000 --env MODEL_PATH=model.joblib --env SCALER_PATH=scaler.joblib placement-predictor:latest

Notes & production tips
- In production, use Gunicorn + nginx, run behind HTTPS, and do input validation and rate limiting.
- Save model metadata/version; consider model registry for multiple versions.
- Add monitoring/logging for predictions and drift detection.
- Ensure you do not ship training data with the app.
