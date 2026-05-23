"""FastAPI application for weather forecast inference."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ForecastPayload(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    weather_data: dict[str, Any] | None = None
    include_input: bool = False


class ForecastResult(BaseModel):
    status: str
    city: str
    forecast: str
    backend: str
    generated_at: str
    input_prompt: str | None = None


def _prompt_from_payload(payload: ForecastPayload) -> str:
    if payload.weather_data:
        data = payload.weather_data
        return (
            f"Weather data for {payload.city} on {data.get('datetime', 'current time')}:\n"
            f"- Temperature (C): {data.get('temperature', 'unknown')}\n"
            f"- Humidity (%): {data.get('humidity', 'unknown')}\n"
            f"- Wind speed (km/h): {data.get('wind_speed', 'unknown')}\n"
            f"- Pressure (hPa): {data.get('pressure', 'unknown')}\n"
            f"- Precipitation probability: {data.get('precipitation_probability', 'unknown')}\n\n"
            "Generate a forecast bulletin:"
        )

    from weather_cli import build_weather_prompt

    return build_weather_prompt(payload.city)


def _fallback_forecast(payload: ForecastPayload) -> str:
    if not payload.weather_data:
        return (
            f"{payload.city} forecast is unavailable until a llama.cpp server "
            "or explicit weather_data payload is provided."
        )

    data = payload.weather_data
    temp = data.get("temperature", "the current range")
    wind = data.get("wind_speed", "local")
    precip = data.get("precipitation_probability", "unknown")
    return (
        f"{payload.city} should see temperatures near {temp}C, winds around "
        f"{wind} km/h, and precipitation probability near {precip}."
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Weather Forecasting LoRA API",
        version="0.1.0",
        description="HTTP wrapper for the local weather forecasting GGUF path.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        server_url = os.environ.get("WEATHER_LLAMA_SERVER_URL")
        return {
            "status": "healthy",
            "backend": "llama.cpp" if server_url else "fallback",
            "llama_server_url": server_url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/forecast", response_model=ForecastResult)
    def forecast(payload: ForecastPayload) -> ForecastResult:
        prompt = _prompt_from_payload(payload)
        server_url = os.environ.get("WEATHER_LLAMA_SERVER_URL")
        backend = "fallback"

        if server_url:
            backend = "llama.cpp"
            try:
                response = requests.post(
                    f"{server_url.rstrip('/')}/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": 80,
                        "temperature": 0.25,
                        "repeat_penalty": 1.25,
                        "stop": ["</s>", "<|user|>", "<|system|>"],
                    },
                    timeout=240,
                )
                response.raise_for_status()
                text = response.json().get("content", "").strip()
                if not text:
                    raise ValueError("llama.cpp returned an empty forecast")
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        else:
            text = _fallback_forecast(payload)

        return ForecastResult(
            status="success",
            city=payload.city,
            forecast=text,
            backend=backend,
            generated_at=datetime.now(timezone.utc).isoformat(),
            input_prompt=prompt if payload.include_input else None,
        )

    return app


app = create_app()
