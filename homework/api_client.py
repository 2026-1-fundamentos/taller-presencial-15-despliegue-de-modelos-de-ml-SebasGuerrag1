"""Cliente Python para consultar el servicio API de pronóstico."""

import json

import requests

API_URL = "http://127.0.0.1:5000/predict"

SAMPLE_HOUSE = {
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1.0,
    "waterfront": 0,
    "condition": 4,
}


def get_prediction(house_data: dict) -> float:
    """Consulta el servicio API y retorna el precio pronosticado."""
    response = requests.post(API_URL, json=house_data, timeout=10)
    response.raise_for_status()
    return float(response.json()["price"])


def main() -> None:
    """Ejemplo de consulta al servicio API."""
    price = get_prediction(SAMPLE_HOUSE)
    print("Datos enviados:")
    print(json.dumps(SAMPLE_HOUSE, indent=2))
    print(f"\nPrecio pronosticado: ${price:,.2f}")


if __name__ == "__main__":
    main()