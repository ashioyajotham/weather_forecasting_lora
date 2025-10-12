"""
Weather Forecasting with LoRA Fine-tuning
==========================================

A comprehensive implementation of weather forecasting using LoRA (Low-Rank Adaptation)
fine-tuning on Large Language Models, following Schulman et al. (2025) methodology.

This project implements:
1. Numerical weather data → Text forecast mapping
2. Supervised Fine-Tuning (SFT) with LoRA adapters
3. Reinforcement Learning from Human Feedback (RLHF) with PPO
4. Evaluation and monitoring systems

Architecture:
- Base Model: Mistral-7B or LLaMA-3-8B
- Data Sources: ERA5, Open-Meteo, NOAA
- Training: LoRA adapters with frozen base weights
- Deployment: Modular adapters for different forecasting domains
"""

from .data import WeatherDataCollector, WeatherPreprocessor
from .models import WeatherForecasterLoRA, LoRATrainer
from .evaluation import WeatherEvaluator, MetricsCalculator
from .rl import PPOTrainerWeather, WeatherRewardModel
from .inference import WeatherInference, ForecastAPI

__version__ = "1.0.0"
__author__ = "Ashioya Jotham Victor"
__email__ = "victorashioya960@gmail.com"

__all__ = [
    "WeatherDataCollector",
    "WeatherPreprocessor", 
    "WeatherForecasterLoRA",
    "LoRATrainer",
    "WeatherEvaluator",
    "MetricsCalculator",
    "PPOTrainerWeather",
    "WeatherRewardModel",
    "WeatherInference",
    "ForecastAPI",
]