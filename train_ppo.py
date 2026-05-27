"""PPO readiness and guarded training entrypoint.

Default mode is a dry run because PPO is expensive and should not start by
accident on a laptop CPU. Use --run-training after confirming GPU memory,
dataset size, and TRL compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("WANDB_DISABLED", "true")


def load_samples(path: Path, limit: int) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data[:limit]


def dry_run(data_path: Path, limit: int) -> None:
    from scripts.ppo_smoke import check_model_artifacts, test_reward_model, test_trl_imports
    from src.rl.ppo_trainer import WeatherRewardModel

    samples = load_samples(data_path, limit)
    reward_model = WeatherRewardModel()
    rewards = [
        reward_model.calculate_composite_reward(
            sample.get("target") or sample.get("response", "")
        ).total_reward
        for sample in samples
    ]

    test_reward_model()
    test_trl_imports()
    check_model_artifacts()
    avg_reward = sum(rewards) / max(len(rewards), 1)
    print(f"OK: PPO dry run scored {len(samples)} samples, avg_reward={avg_reward:.3f}")
    print("OK: use --run-training to start the guarded PPO training path")


def run_training(data_path: Path, limit: int, output_dir: Path) -> None:
    """Start the legacy PPO trainer only after explicit opt-in."""
    import torch

    from src.rl.ppo_trainer import PPOTrainerWeather, WeatherRewardModel

    if not torch.cuda.is_available():
        raise SystemExit(
            "PPO training requires a CUDA GPU for TRL value-head models. "
            "Local CPU/disk offload can run the dry run, but not PPO training."
        )

    model_path = ROOT / "models" / "weather-lora-peft" / "lora_adapter"
    if not (model_path / "adapter_config.json").exists():
        raise SystemExit(
            f"Missing trainable LoRA adapter for PPO: {model_path}. "
            "Run/import the TinyLlama LoRA adapter before PPO. "
            "PPO on the merged full model is intentionally disabled because it produced NaN policy losses."
        )

    samples = load_samples(data_path, limit)
    trainer = PPOTrainerWeather(
        model_path=str(model_path),
        reward_model=WeatherRewardModel(),
        config_path=str(ROOT / "config" / "ppo_config.yaml"),
    )
    trainer.train(
        training_data=samples,
        num_epochs=1,
        save_every=25,
        output_dir=str(output_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "train.json")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("WEATHER_PPO_MAX_SAMPLES", "16")))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "weather-lora-ppo")
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    if args.run_training:
        run_training(args.data, args.limit, args.output_dir)
    else:
        dry_run(args.data, args.limit)


if __name__ == "__main__":
    main()
