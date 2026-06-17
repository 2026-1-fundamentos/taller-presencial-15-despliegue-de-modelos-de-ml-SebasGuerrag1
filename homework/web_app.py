"""Aplicación web Flask para pronosticar precios de casas."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

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


def parse_form_data(form) -> pd.DataFrame:
    """Convierte los datos del formulario en un DataFrame para el modelo."""
    data = {
        "bedrooms": int(form["bedrooms"]),
        "bathrooms": float(form["bathrooms"]),
        "sqft_living": int(form["sqft_living"]),
        "sqft_lot": int(form["sqft_lot"]),
        "floors": float(form["floors"]),
        "waterfront": int(form["waterfront"]),
        "condition": int(form["condition"]),
    }
    return pd.DataFrame([data])


@app.route("/", methods=["GET", "POST"])
def index():
    """Muestra el formulario y retorna el precio pronosticado."""
    prediction = None
    if request.method == "POST":
        input_frame = parse_form_data(request.form)
        prediction = float(model.predict(input_frame)[0])

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)