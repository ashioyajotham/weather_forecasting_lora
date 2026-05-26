"""
PPO and RLHF regression tests.
"""

import pytest
import torch

from src.rl.ppo_trainer import PPOTrainerWeather, WeatherRewardModel


class ValueHeadLikeModel(torch.nn.Module):
    """Minimal TRL-style wrapper with parameters but no .device attribute."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def test_ppo_trainer_resolves_device_without_model_device_attribute():
    trainer = PPOTrainerWeather(
        model_path="unused",
        reward_model=WeatherRewardModel(),
    )
    trainer.model = ValueHeadLikeModel()

    assert trainer._model_device() == trainer.model.weight.device


def test_ppo_training_fails_when_all_steps_fail(temp_dir):
    trainer = PPOTrainerWeather(
        model_path="unused",
        reward_model=WeatherRewardModel(),
    )
    trainer.model = ValueHeadLikeModel()
    trainer.ppo_trainer = object()

    def failing_train_step(batch, observed_weather=None):
        raise AttributeError("synthetic training failure")

    trainer.train_step = failing_train_step

    with pytest.raises(RuntimeError, match="zero successful steps"):
        trainer.train(
            training_data=[{"input": "Weather data\nGenerate a forecast bulletin:"}],
            num_epochs=1,
            output_dir=str(temp_dir / "ppo-zero-success"),
        )
