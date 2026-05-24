"""Prepare local processed data for smoke training runs.

Fresh Colab clones do not include ignored data/processed JSON files. This
script validates existing processed data when present, or creates a small
deterministic synthetic fallback dataset for plumbing and notebook smoke runs.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"

LOCATIONS = [
    ("Nairobi", 21.0, 27.0, 65.0, 12.0, 1015.0, 0.12),
    ("London", 10.0, 18.0, 82.0, 22.0, 1008.0, 0.55),
    ("Phoenix", 30.0, 39.0, 18.0, 10.0, 1011.0, 0.02),
    ("Mumbai", 27.0, 31.0, 84.0, 20.0, 1005.0, 0.62),
    ("Toronto", -6.0, 4.0, 58.0, 18.0, 1022.0, 0.10),
    ("Sydney", 18.0, 25.0, 66.0, 17.0, 1018.0, 0.20),
]


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list")
    return data


def _is_usable(path: Path, minimum: int) -> bool:
    if not path.is_file():
        return False
    try:
        data = _load_json(path)
    except Exception:
        return False
    if len(data) < minimum:
        return False
    sample = data[0] if data else {}
    return {"input", "target"} <= set(sample)


def processed_data_ready(processed_dir: Path, minimums: dict[str, int]) -> bool:
    return all(_is_usable(processed_dir / name, minimum) for name, minimum in minimums.items())


def _trend(values: list[float], step: float) -> list[float]:
    return [round(value + (index * step), 1) for index, value in enumerate(values)]


def _forecast_text(
    min_temp: float,
    max_temp: float,
    avg_humidity: float,
    max_wind: float,
    avg_precip: float,
) -> str:
    avg_temp = (min_temp + max_temp) / 2
    if avg_temp < 0:
        temp_word = "very cold"
    elif avg_temp < 10:
        temp_word = "cold"
    elif avg_temp < 20:
        temp_word = "cool"
    elif avg_temp < 30:
        temp_word = "mild"
    else:
        temp_word = "hot"

    if avg_precip >= 0.55:
        condition = "rain likely"
    elif avg_precip >= 0.30:
        condition = "showers possible"
    elif avg_humidity >= 80:
        condition = "cloudy"
    else:
        condition = "mainly dry"

    if max_wind >= 28:
        wind_text = f"windy with gusts near {max_wind:.0f} km/h"
    elif max_wind >= 18:
        wind_text = f"breezy with winds around {max_wind:.0f} km/h"
    else:
        wind_text = f"light winds near {max_wind:.0f} km/h"

    return (
        f"{condition.capitalize()} conditions expected with {temp_word} "
        f"temperatures around {min_temp:.0f}-{max_temp:.0f}C and {wind_text}."
    )


def make_sample(index: int, rng: random.Random) -> dict[str, str]:
    name, low, high, humidity, wind, pressure, precip = LOCATIONS[index % len(LOCATIONS)]
    phase = index // len(LOCATIONS)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=phase * 3)

    base_temp = rng.uniform(low, high)
    temp_step = rng.uniform(-0.4, 0.6)
    temperatures = _trend([base_temp + rng.uniform(-0.5, 0.5) for _ in range(4)], temp_step)
    humidities = [round(max(10, min(98, humidity + rng.uniform(-6, 6))), 0) for _ in range(4)]
    winds = [round(max(1, wind + rng.uniform(-4, 5)), 1) for _ in range(4)]
    pressures = [round(pressure + rng.uniform(-3, 3), 1) for _ in range(4)]
    precips = [round(max(0.0, min(0.95, precip + rng.uniform(-0.08, 0.12))), 2) for _ in range(4)]

    prompt = (
        f"Weather data for {name} on {start.strftime('%Y-%m-%d %H:%M UTC')}:\n"
        f"- Temperature (C): {', '.join(f'{value:.1f}' for value in temperatures)}\n"
        f"- Humidity (%): {', '.join(f'{value:.0f}' for value in humidities)}\n"
        f"- Wind speed (km/h): {', '.join(f'{value:.1f}' for value in winds)}\n"
        f"- Pressure (hPa): {', '.join(f'{value:.1f}' for value in pressures)}\n"
        f"- Precipitation probability: {', '.join(f'{value:.2f}' for value in precips)}\n\n"
        "Generate a forecast bulletin:"
    )
    target = _forecast_text(
        min(temperatures),
        max(temperatures),
        sum(humidities) / len(humidities),
        max(winds),
        sum(precips) / len(precips),
    )

    return {
        "input": prompt,
        "target": target,
        "prompt": prompt,
        "response": target,
        "location": name,
        "datetime": start.strftime("%Y-%m-%d %H:%M UTC"),
    }


def generate_synthetic_dataset(
    processed_dir: Path,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    total = train_count + val_count + test_count
    data = [make_sample(index, rng) for index in range(total)]

    splits = {
        "train.json": data[:train_count],
        "val.json": data[train_count : train_count + val_count],
        "test.json": data[train_count + val_count :],
    }
    for filename, rows in splits.items():
        with (processed_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)

    stats = {
        "source": "synthetic_colab_fallback",
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train": train_count,
        "val": val_count,
        "test": test_count,
        "locations": [location[0] for location in LOCATIONS],
    }
    with (processed_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return {"train": train_count, "val": val_count, "test": test_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--min-train", type=int, default=240)
    parser.add_argument("--min-val", type=int, default=30)
    parser.add_argument("--min-test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-if-missing", action="store_true")
    args = parser.parse_args()

    minimums = {
        "train.json": args.min_train,
        "val.json": args.min_val,
        "test.json": args.min_test,
    }
    processed_dir = args.processed_dir.resolve()

    if processed_data_ready(processed_dir, minimums):
        print(f"OK: processed data already usable in {processed_dir}")
        return

    if not args.synthetic_if_missing:
        raise SystemExit(
            "Processed data is missing or too small. Re-run with --synthetic-if-missing "
            "for a deterministic smoke-training fallback."
        )

    counts = generate_synthetic_dataset(
        processed_dir=processed_dir,
        train_count=args.min_train,
        val_count=args.min_val,
        test_count=args.min_test,
        seed=args.seed,
    )
    print(
        "OK: generated synthetic processed data "
        f"train={counts['train']} val={counts['val']} test={counts['test']} in {processed_dir}"
    )


if __name__ == "__main__":
    main()
