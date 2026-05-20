"""Smoke test for evaluation metrics without requiring model inference."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import ForecastPrediction, MetricsCalculator


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"OK: {message}")


def main() -> None:
    calculator = MetricsCalculator()

    generated = "Partly cloudy with temperatures around 22-25°C. Light winds up to 15 km/h. No rain expected."
    reference = "Partly cloudy skies with highs near 24°C. Breezy conditions with winds to 18 km/h. Dry conditions."

    bleu = calculator.calculate_bleu_score(generated, reference)
    rouge = calculator.calculate_rouge_scores(generated, reference)
    if not 0.0 <= bleu <= 1.0:
        fail(f"BLEU outside expected range: {bleu}")
    if rouge["rouge1_f"] <= 0.0:
        fail(f"ROUGE fallback returned no overlap: {rouge}")
    ok(f"text metrics: bleu={bleu:.3f}, rouge1={rouge['rouge1_f']:.3f}")

    prediction = ForecastPrediction(
        input_prompt="Weather data for test",
        generated_forecast=generated,
        reference_forecast=reference,
        location="Test",
        datetime="2026-05-20",
    )
    metrics = calculator.calculate_all_metrics([prediction])
    if not 0.0 <= metrics.overall_score <= 1.0:
        fail(f"overall score outside expected range: {metrics.overall_score}")
    ok(f"overall score: {metrics.overall_score:.3f}")


if __name__ == "__main__":
    main()
