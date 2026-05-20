"""Project smoke checks that avoid model loading and network calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"OK: {message}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"{label} missing: {path.relative_to(ROOT)}")
    ok(f"{label}: {path.relative_to(ROOT)}")


def check_imports() -> None:
    sys.path.insert(0, str(ROOT))

    import src  # noqa: F401
    from src.data import ForecastPrompt, WeatherDataCollector, WeatherPreprocessor
    from src.evaluation import MetricsCalculator, WeatherEvaluator
    from src.inference import ForecastRequest, WeatherInference

    _ = (
        ForecastPrompt,
        WeatherDataCollector,
        WeatherPreprocessor,
        MetricsCalculator,
        WeatherEvaluator,
        ForecastRequest,
        WeatherInference,
    )
    ok("core package imports")


def check_processed_data() -> None:
    expected_min_counts = {
        "train.json": 1,
        "val.json": 1,
        "test.json": 1,
    }

    processed_dir = ROOT / "data" / "processed"
    for filename, minimum in expected_min_counts.items():
        path = processed_dir / filename
        require_file(path, f"processed data {filename}")
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or len(data) < minimum:
            fail(f"{path.relative_to(ROOT)} has no usable samples")

        sample = data[0]
        missing = {"input", "target"} - set(sample)
        if missing:
            fail(f"{path.relative_to(ROOT)} sample missing keys: {sorted(missing)}")
        ok(f"{filename} samples: {len(data)}")


def check_configs_and_scripts() -> None:
    for relative in [
        "config/sft_config.yaml",
        "config/ppo_config.yaml",
        "train_lora_peft.py",
        "merge_lora.py",
        "weather_cli.py",
    ]:
        require_file(ROOT / relative, "required project file")


def check_cli_artifacts() -> None:
    model = ROOT / "models" / "gguf" / "weather-tinyllama.gguf"
    server = ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"

    if model.is_file() and server.is_file():
        ok("CLI artifacts are present")
        return

    print("WARN: CLI artifacts are incomplete; weather_cli.py will not run yet.")
    if not model.is_file():
        print(f"WARN: missing {model.relative_to(ROOT)}")
    if not server.is_file():
        print(f"WARN: missing {server.relative_to(ROOT)}")


def main() -> None:
    check_imports()
    check_processed_data()
    check_configs_and_scripts()
    check_cli_artifacts()
    ok("smoke check completed")


if __name__ == "__main__":
    main()
