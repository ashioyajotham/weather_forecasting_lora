"""Tests for API app wiring and pinned generation-quality fixtures."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from scripts.generation_quality_eval import load_cases, score_case
from src.inference.api import app


def test_fastapi_health_and_fallback_forecast():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    response = client.post(
        "/forecast",
        json={
            "city": "Nairobi",
            "weather_data": {
                "temperature": "23-25",
                "wind_speed": "10-15",
                "precipitation_probability": "0.10",
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "fallback"
    assert "Nairobi" in payload["forecast"]


def test_generation_quality_fixture_scores_reference_forecasts():
    eval_path = Path("data/eval/generation_quality.json")
    cases = load_cases(eval_path)

    assert len(cases) >= 5
    for case in cases:
        result = score_case(case, case["reference_forecast"])
        assert result["passed"], json.dumps(result, indent=2)
