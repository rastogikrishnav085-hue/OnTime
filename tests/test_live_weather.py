"""Live weather endpoint contract tests; provider calls are mocked, so tests never depend on the internet."""
import os
import sys
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_live_weather_endpoint_returns_source_and_not_model_feature(monkeypatch, tmp_path):
    from backend.app import main
    main.set_db_path(str(tmp_path / "weather-board.db"))
    expected = {"available": True, "live": True, "source": "Open-Meteo forecast API", "current": {"temperature_2m": 24},
                "daily": {"time": ["2026-10-24"]}, "training_use": False, "note": "Live only"}
    monkeypatch.setattr(main.wx, "forecast", lambda latitude, longitude: expected)
    with TestClient(main.app) as client:
        r = client.get("/api/weather/forecast", params={"latitude": -23.55, "longitude": -46.63})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "Open-Meteo forecast API"
    assert body["training_use"] is False
    assert body["live"] is True


def test_live_weather_rejects_invalid_coordinates(tmp_path):
    from backend.app import main
    main.set_db_path(str(tmp_path / "weather-invalid-board.db"))
    with TestClient(main.app) as client:
        r = client.get("/api/weather/forecast", params={"latitude": 91, "longitude": 0})
    assert r.status_code == 422


def test_live_weather_provider_failure_is_not_replaced_with_synthetic_data(monkeypatch, tmp_path):
    from backend.app import main
    main.set_db_path(str(tmp_path / "weather-down-board.db"))
    def unavailable(latitude, longitude):
        raise main.wx.WeatherUnavailable("provider down; no synthetic substitution")
    monkeypatch.setattr(main.wx, "forecast", unavailable)
    with TestClient(main.app) as client:
        r = client.get("/api/weather/forecast", params={"latitude": -23.55, "longitude": -46.63})
    assert r.status_code == 503
    assert "no synthetic substitution" in r.json()["detail"]
