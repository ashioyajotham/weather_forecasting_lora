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
        self.generate_calls = []

    def generate(self, input_ids, **kwargs):
        self.generate_calls.append(kwargs)
        response = torch.tensor([[10, 11]], device=input_ids.device)
        return torch.cat([input_ids, response], dim=1)


class MinimalTokenizer:
    pad_token_id = 0

    def encode(self, prompt, return_tensors=None):
        return torch.tensor([[1, 2, 3]])

    def decode(self, response_tensor, skip_special_tokens=True):
        return "Forecast calls for rain with temperatures around 22-24C."


class CapturingPPOTrainer:
    def __init__(self):
        self.calls = []

    def step(self, queries, responses, scores):
        self.calls.append((queries, responses, scores))
        assert isinstance(queries, list)
        assert isinstance(responses, list)
        assert isinstance(scores, list)
        assert all(isinstance(query, torch.Tensor) for query in queries)
        assert all(isinstance(response, torch.Tensor) for response in responses)
        assert all(isinstance(score, torch.Tensor) for score in scores)
        assert all(score.dim() == 0 for score in scores)
        assert len(queries) == len(responses) == len(scores) == 2
        return {"ppo/returns/mean": torch.stack(scores).mean().item()}


def test_ppo_trainer_resolves_device_without_model_device_attribute():
    trainer = PPOTrainerWeather(
        model_path="unused",
        reward_model=WeatherRewardModel(),
    )
    trainer.model = ValueHeadLikeModel()

    assert trainer._model_device() == trainer.model.weight.device


def test_ppo_train_step_passes_list_of_scalar_score_tensors():
    trainer = PPOTrainerWeather(
        model_path="unused",
        reward_model=WeatherRewardModel(),
    )
    trainer.model = ValueHeadLikeModel()
    trainer.tokenizer = MinimalTokenizer()
    trainer.ppo_trainer = CapturingPPOTrainer()

    stats = trainer.train_step(
        [
            {"input": "Weather data for Nairobi\nGenerate a forecast bulletin:"},
            {"input": "Weather data for Kisumu\nGenerate a forecast bulletin:"},
        ]
    )

    assert "ppo/returns/mean" in stats
    assert len(trainer.ppo_trainer.calls) == 1
    assert len(trainer.model.generate_calls) == 2
    for generate_kwargs in trainer.model.generate_calls:
        assert "attention_mask" in generate_kwargs
        assert generate_kwargs["max_new_tokens"] == trainer.config["max_new_tokens"]
        assert generate_kwargs["temperature"] == trainer.config["temperature"]
        assert generate_kwargs["top_p"] == trainer.config["top_p"]
        assert generate_kwargs["top_k"] == trainer.config["top_k"]
        assert generate_kwargs["remove_invalid_values"] is True
        assert generate_kwargs["renormalize_logits"] is True


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


def test_ppo_training_reraises_fatal_cuda_errors(temp_dir):
    trainer = PPOTrainerWeather(
        model_path="unused",
        reward_model=WeatherRewardModel(),
    )
    trainer.model = ValueHeadLikeModel()
    trainer.ppo_trainer = object()

    def failing_train_step(batch, observed_weather=None):
        raise RuntimeError("CUDA error: device-side assert triggered")

    trainer.train_step = failing_train_step

    with pytest.raises(RuntimeError, match="device-side assert"):
        trainer.train(
            training_data=[{"input": "Weather data\nGenerate a forecast bulletin:"}],
            num_epochs=1,
            output_dir=str(temp_dir / "ppo-fatal"),
        )
