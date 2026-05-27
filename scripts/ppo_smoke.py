"""Smoke check for PPO reward-model infrastructure.

This does not run PPO training. It validates the reward model, TRL imports, and
that the local SFT/merged model artifacts exist when present.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from importlib.metadata import version
from inspect import signature
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_reward_model() -> None:
    """Check reward model scoring on representative forecasts."""
    from src.rl.ppo_trainer import WeatherRewardModel

    reward_model = WeatherRewardModel(
        accuracy_weight=0.4,
        style_weight=0.2,
        calibration_weight=0.2,
        consistency_weight=0.2,
    )

    cases = [
        (
            "good forecast",
            "Partly cloudy with temperatures around 22-25C. Light winds up to 15 km/h. Showers possible by evening.",
            {"rain": True, "temperature": 23.5},
        ),
        (
            "vague forecast",
            "Weather expected.",
            {"rain": False, "temperature": 20.0},
        ),
    ]

    for name, forecast, observed in cases:
        rewards = reward_model.calculate_composite_reward(forecast, observed)
        logger.info("%s total reward: %.3f", name, rewards.total_reward)
        if not 0.0 <= rewards.total_reward <= 1.0:
            raise SystemExit(f"Reward out of range for {name}: {rewards.total_reward}")


def test_trl_imports() -> None:
    """Verify TRL PPO dependencies expose the classic manual-step API."""
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    _ = (AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer)
    ppo_config_params = signature(PPOConfig).parameters
    ppo_trainer_params = signature(PPOTrainer).parameters
    project_ppo_config_keys = {
        "batch_size",
        "forward_batch_size",
        "learning_rate",
        "mini_batch_size",
        "gradient_accumulation_steps",
        "ppo_epochs",
        "kl_penalty",
        "init_kl_coef",
        "target_kl",
        "cliprange",
        "vf_coef",
        "max_grad_norm",
        "early_stopping",
        "ratio_threshold",
        "score_clip",
        "use_score_scaling",
        "use_score_norm",
        "whiten_rewards",
        "seed",
        "log_with",
    }

    missing = []
    for key in sorted(project_ppo_config_keys):
        if key not in ppo_config_params:
            missing.append(f"PPOConfig.{key}")
    if "config" not in ppo_trainer_params:
        missing.append("PPOTrainer(config=...)")
    if "tokenizer" not in ppo_trainer_params:
        missing.append("PPOTrainer(tokenizer=...)")
    if not hasattr(PPOTrainer, "step"):
        missing.append("PPOTrainer.step")

    if missing:
        installed = version("trl")
        raise SystemExit(
            "Installed TRL is incompatible with this project's manual PPO loop. "
            f"Installed trl=={installed}; missing {', '.join(missing)}. "
            "Install trl==0.11.4 or update the PPO trainer to the newer TRL API."
        )

    logger.info("TRL PPO classic API OK: trl==%s", version("trl"))


def test_ppo_config_types() -> None:
    """Verify PPO YAML/default config is normalized before TRL receives it."""
    from src.rl.ppo_trainer import PPOTrainerWeather, WeatherRewardModel

    trainer = PPOTrainerWeather(
        model_path=str(ROOT / "models" / "weather-merged"),
        reward_model=WeatherRewardModel(),
        config_path=str(ROOT / "config" / "ppo_config.yaml"),
    )
    config = trainer._load_config()
    int_fields = {
        "batch_size",
        "forward_batch_size",
        "mini_batch_size",
        "gradient_accumulation_steps",
        "ppo_epochs",
        "seed",
    }
    float_fields = {
        "learning_rate",
        "init_kl_coef",
        "target_kl",
        "cliprange",
        "vf_coef",
        "max_grad_norm",
        "ratio_threshold",
        "score_clip",
    }

    bad_fields = [
        f"{field}={config[field]!r} ({type(config[field]).__name__})"
        for field in sorted(int_fields)
        if not isinstance(config[field], int)
    ]
    bad_fields.extend(
        f"{field}={config[field]!r} ({type(config[field]).__name__})"
        for field in sorted(float_fields)
        if not isinstance(config[field], float)
    )
    if not isinstance(config["kl_penalty"], str):
        bad_fields.append(
            f"kl_penalty={config['kl_penalty']!r} ({type(config['kl_penalty']).__name__})"
        )
    for field in ["early_stopping", "use_score_scaling", "use_score_norm", "whiten_rewards"]:
        if not isinstance(config[field], bool):
            bad_fields.append(f"{field}={config[field]!r} ({type(config[field]).__name__})")

    if bad_fields:
        raise SystemExit("PPO config has unnormalized field types: " + ", ".join(bad_fields))

    logger.info("PPO config numeric types OK")


def check_model_artifacts() -> None:
    """Report current local model artifact state."""
    paths = [
        ROOT / "models" / "weather-lora-peft" / "lora_adapter",
        ROOT / "models" / "weather-merged",
        ROOT / "models" / "gguf" / "weather-tinyllama.gguf",
    ]
    for path in paths:
        if path.exists():
            logger.info("found artifact: %s", path.relative_to(ROOT))
        else:
            logger.warning("missing artifact: %s", path.relative_to(ROOT))


def test_real_forecasts() -> None:
    """Score a few existing target forecasts from the test split."""
    from src.rl.ppo_trainer import WeatherRewardModel

    test_file = ROOT / "data" / "processed" / "test.json"
    if not test_file.exists():
        logger.warning("test data missing: %s", test_file.relative_to(ROOT))
        return

    with test_file.open(encoding="utf-8") as handle:
        data = json.load(handle)

    reward_model = WeatherRewardModel()
    rewards = []
    for sample in data[:5]:
        forecast = sample.get("response", sample.get("target", ""))
        score = reward_model.calculate_composite_reward(forecast).total_reward
        rewards.append(score)

    average = sum(rewards) / len(rewards)
    logger.info("average reward over %d target forecasts: %.3f", len(rewards), average)


def main() -> None:
    test_reward_model()
    test_trl_imports()
    test_ppo_config_types()
    check_model_artifacts()
    test_real_forecasts()
    logger.info("PPO smoke check completed")


if __name__ == "__main__":
    main()
