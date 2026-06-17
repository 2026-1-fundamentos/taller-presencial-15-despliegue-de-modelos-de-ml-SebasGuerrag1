"""Entrenamiento del modelo de pronóstico de precios de casas."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "files" / "input" / "house_data.csv"
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


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carga y prepara los datos de entrenamiento."""
    data = pd.read_csv(DATA_PATH)
    data["floors"] = pd.to_numeric(data["floors"], errors="coerce")
    x = data[FEATURES]
    y = data["price"]
    return x, y


def train_model() -> Pipeline:
    """Entrena y retorna el pipeline del modelo."""
    x, y = load_data()
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    return model


def save_model(model: Pipeline) -> None:
    """Persiste el modelo entrenado en disco."""
    joblib.dump(model, MODEL_PATH)


def main() -> None:
    """Entrena el modelo y lo guarda."""
    model = train_model()
    save_model(model)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()