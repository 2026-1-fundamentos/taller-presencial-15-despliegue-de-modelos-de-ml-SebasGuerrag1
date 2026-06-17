"""Servicio API para pronosticar precios de casas."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_predictor.pkl"

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def build_input_frame(data: dict) -> pd.DataFrame:
    """Construye el DataFrame de entrada a partir del JSON recibido."""
    return pd.DataFrame([{feature: data[feature] for feature in FEATURES}])


@app.route("/predict", methods=["POST"])
def predict():
    """Retorna el pronóstico del precio para una casa."""
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Se requiere un cuerpo JSON"}), 400

    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        return jsonify({"error": f"Faltan campos: {', '.join(missing)}"}), 400

    input_frame = build_input_frame(data)
    prediction = float(model.predict(input_frame)[0])
    return jsonify({"price": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)