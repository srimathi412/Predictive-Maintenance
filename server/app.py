from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load trained model
# -------------------------
with open("../model/rul_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------
# Load scaler
# -------------------------
with open("../model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------
# Load NASA dataset
# -------------------------
data_df = pd.read_csv(
    "../data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

columns = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

data_df.columns = columns

# -------------------------
# Home API route
# -------------------------
@app.route("/")
def home():
    return "Predictive Maintenance API is running"

# -------------------------
# UI route
# -------------------------
@app.route("/ui")
def ui():
    return render_template("index.html")

# -------------------------
# Prediction API
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if "engine_id" not in data:
        return jsonify({"error": "engine_id is required"}), 400

    engine_id = int(data["engine_id"])

    engine_data = data_df[data_df["engine_id"] == engine_id]

    if engine_data.empty:
        return jsonify({"error": "Invalid engine_id"}), 400

    # Take latest cycle
    latest_row = engine_data.iloc[len(engine_data) // 2]

    # Prepare features
    features = latest_row.drop(["engine_id", "cycle"]).values.reshape(1, -1)
    features = scaler.transform(features)

    # Predict RUL
    predicted_rul = model.predict(features)[0]

    return jsonify({
        "engine_id": engine_id,
        "predicted_RUL": round(float(predicted_rul), 2)
    })

# -------------------------
# Run Flask app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
