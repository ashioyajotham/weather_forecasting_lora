"""Pinned generation-quality evaluation for weather forecasts.

The fixture is intentionally small and stable. It catches regressions that the
model smoke tests miss: prompt leakage, table fragments, missing weather
content, and low reward/style scores.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("WANDB_DISABLED", "true")


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"Eval set must be a non-empty list: {path}")
    return data


def load_predictions(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}

    predictions: dict[str, str] = {}
    for item in data:
        case_id = item.get("id")
        generated = item.get("generated_forecast") or item.get("forecast")
        if case_id and generated:
            predictions[str(case_id)] = str(generated)
    return predictions


def score_case(case: dict[str, Any], generated: str) -> dict[str, Any]:
    from src.evaluation import ForecastPrediction, MetricsCalculator
    from src.rl.ppo_trainer import WeatherRewardModel

    calculator = MetricsCalculator()
    reward_model = WeatherRewardModel()
    reference = case["reference_forecast"]

    prediction = ForecastPrediction(
        input_prompt=case["input"],
        generated_forecast=generated,
        reference_forecast=reference,
        location=case["location"],
        datetime="pinned",
    )
    metrics = calculator.calculate_all_metrics([prediction])
    reward = reward_model.calculate_composite_reward(
        generated,
        case.get("observed_weather"),
    )

    lower = generated.lower()
    required = [term for term in case.get("required_terms", []) if term.lower() in lower]
    forbidden = [
        pattern
        for pattern in case.get("forbidden_patterns", [])
        if re.search(pattern, generated, flags=re.IGNORECASE)
    ]

    words = generated.split()
    passed = (
        metrics.overall_score >= 0.35
        and reward.total_reward >= 0.55
        and len(required) >= max(1, len(case.get("required_terms", [])) - 1)
        and not forbidden
        and 8 <= len(words) <= 55
    )

    return {
        "id": case["id"],
        "passed": passed,
        "overall_score": metrics.overall_score,
        "reward": reward.total_reward,
        "required_terms_found": required,
        "forbidden_patterns_found": forbidden,
        "word_count": len(words),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=ROOT / "data" / "eval" / "generation_quality.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional JSON predictions keyed by eval id, or list with id/generated_forecast.",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        help="Optional path for JSON report output.",
    )
    args = parser.parse_args()

    cases = load_cases(args.eval_set)
    predictions = load_predictions(args.predictions) if args.predictions else {}

    results = []
    for case in cases:
        generated = predictions.get(case["id"], case["reference_forecast"])
        results.append(score_case(case, generated))

    pass_count = sum(1 for result in results if result["passed"])
    report = {
        "eval_set": str(args.eval_set.relative_to(ROOT)),
        "cases": len(results),
        "passed": pass_count,
        "pass_rate": pass_count / len(results),
        "results": results,
    }

    print(json.dumps(report, indent=2))

    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if pass_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
