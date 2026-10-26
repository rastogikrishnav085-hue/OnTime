"""Live weather lookup via Open-Meteo. This is operational context only, not Olist training data."""
from __future__ import annotations
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

class WeatherUnavailable(RuntimeError):
    pass

def forecast(latitude: float, longitude: float) -> dict:
    if not (-90 <= float(latitude) <= 90 and -180 <= float(longitude) <= 180):
        raise ValueError("Latitude must be between -90 and 90 and longitude between -180 and 180.")
    params = urlencode({
        "latitude": float(latitude), "longitude": float(longitude),
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto", "forecast_days": 3,
    })
    url = "https://api.open-meteo.com/v1/forecast?" + params
    req = Request(url, headers={"User-Agent": "OnTime-Delivery-Dashboard/1.0"})
    try:
        with urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise WeatherUnavailable(f"Live weather provider is unavailable: {exc.__class__.__name__}. Synthetic tracker data has not been substituted.") from exc
    return {
        "available": True, "live": True, "source": "Open-Meteo forecast API",
        "latitude": data.get("latitude"), "longitude": data.get("longitude"),
        "timezone": data.get("timezone"), "current": data.get("current"), "daily": data.get("daily"),
        "training_use": False,
        "note": "Live current/forecast weather only. Olist has no historical weather observations; this response is not used to train or retrospectively score historical orders.",
    }
