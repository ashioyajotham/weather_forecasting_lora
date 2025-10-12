"""
Pytest Configuration and Fixtures for Weather Forecasting LoRA Tests
=====================================================================

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, MagicMock


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "base_model": "microsoft/Mistral-7B-v0.1",
        "device": "cpu",  # Use CPU for testing
        "max_length": 128,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "num_epochs": 1,
    }


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_weather_data():
    """Provide sample weather data for testing."""
    return {
        "location": "New York",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "datetime": "2025-10-12 12:00 UTC",
        "temperature": [23.0, 24.0, 22.0, 21.0],
        "humidity": [70, 75, 80, 82],
        "wind_speed": [12.0, 18.0, 20.0, 15.0],
        "pressure": [1008.0, 1007.0, 1006.0, 1005.0],
        "precipitation_probability": [0.1, 0.2, 0.6, 0.7],
    }


@pytest.fixture
def sample_training_data():
    """Provide sample training data."""
    return [
        {
            "input": """Weather data for New York on 2025-10-12 12:00 UTC:
- Temperature (°C): 23.0, 24.0, 22.0, 21.0
- Humidity (%): 70, 75, 80, 82
- Wind speed (km/h): 12.0, 18.0, 20.0, 15.0
- Pressure (hPa): 1008.0, 1007.0, 1006.0, 1005.0
- Precipitation probability: 0.10, 0.20, 0.60, 0.70

Generate a forecast bulletin:""",
            "target": "Afternoon temperatures around 23-24°C with high humidity. Winds increasing to 20 km/h by early evening. Showers likely by evening with precipitation chances above 60%.",
        },
        {
            "input": """Weather data for London on 2025-10-12 09:00 UTC:
- Temperature (°C): 18.0, 19.0, 17.0, 16.0
- Humidity (%): 80, 85, 90, 92
- Wind speed (km/h): 25.0, 28.0, 30.0, 26.0
- Pressure (hPa): 1005.0, 1004.0, 1003.0, 1002.0
- Precipitation probability: 0.50, 0.70, 0.80, 0.75

Generate a forecast bulletin:""",
            "target": "Overcast with temperatures around 17-19°C. Strong winds up to 30 km/h. Rain showers expected with high probability throughout the day.",
        },
    ]


@pytest.fixture
def sample_test_data():
    """Provide sample test data."""
    return [
        {
            "input": """Weather data for Tokyo on 2025-10-12 06:00 UTC:
- Temperature (°C): 25.0, 26.0, 27.0, 26.0
- Humidity (%): 65, 68, 70, 72
- Wind speed (km/h): 10.0, 12.0, 15.0, 13.0
- Pressure (hPa): 1012.0, 1011.0, 1010.0, 1009.0
- Precipitation probability: 0.05, 0.10, 0.15, 0.20

Generate a forecast bulletin:""",
            "target": "Warm temperatures reaching 26-27°C. Light to moderate winds up to 15 km/h. Generally dry conditions with low chance of precipitation.",
            "location": "Tokyo",
            "datetime": "2025-10-12 06:00 UTC",
        }
    ]


@pytest.fixture
def sample_forecast_predictions():
    """Provide sample forecast predictions for evaluation."""
    from src.evaluation import ForecastPrediction
    
    return [
        ForecastPrediction(
            input_prompt="Weather data for New York...",
            generated_forecast="Partly cloudy with temperatures around 22-25°C. Light winds up to 15 km/h. No rain expected.",
            reference_forecast="Partly cloudy skies with highs near 24°C. Breezy conditions with winds to 18 km/h. Dry conditions.",
            location="New York",
            datetime="2025-10-12"
        ),
        ForecastPrediction(
            input_prompt="Weather data for London...",
            generated_forecast="Overcast with showers likely. Temperatures 18-20°C. Winds increasing to 25 km/h.",
            reference_forecast="Cloudy with rain expected. Cool temperatures around 19°C. Windy with gusts to 28 km/h.",
            location="London",
            datetime="2025-10-12"
        ),
    ]


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing without loading actual model."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "</s>"
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    tokenizer.batch_decode.return_value = ["Sample forecast text"]
    tokenizer.decode.return_value = "Sample forecast text"
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model for testing without loading actual weights."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    model.config = MagicMock()
    model.config.vocab_size = 32000
    return model


@pytest.fixture
def mock_lora_model(mock_model, mock_tokenizer):
    """Mock LoRA model for testing."""
    from src.models import WeatherForecasterLoRA
    
    lora_model = Mock(spec=WeatherForecasterLoRA)
    lora_model.tokenizer = mock_tokenizer
    lora_model.peft_model = mock_model
    lora_model.base_model_name = "microsoft/Mistral-7B-v0.1"
    lora_model.device = "cpu"
    
    # Mock methods
    lora_model.generate_forecast.return_value = "Sample forecast: Partly cloudy with temperatures around 23°C."
    lora_model.load_model.return_value = None
    lora_model.save_model.return_value = None
    
    return lora_model


# ============================================================================
# LoRA Configuration Fixtures
# ============================================================================

@pytest.fixture
def lora_config():
    """Provide LoRA configuration for testing."""
    return {
        "r": 16,  # Smaller rank for faster testing
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }


# ============================================================================
# Evaluation Fixtures
# ============================================================================

@pytest.fixture
def sample_metrics():
    """Provide sample evaluation metrics."""
    from src.evaluation import EvaluationMetrics
    
    return EvaluationMetrics(
        bleu_score=0.45,
        rouge_1_f=0.52,
        rouge_2_f=0.38,
        rouge_l_f=0.48,
        rain_accuracy=0.85,
        temperature_mae=1.2,
        wind_speed_mae=2.5,
        categorical_accuracy=0.80,
        brier_score=0.15,
        reliability=0.90,
        resolution=0.85,
        readability_score=0.75,
        length_similarity=0.88,
        vocabulary_diversity=0.70,
        overall_score=0.68,
        confidence_interval=(0.65, 0.71)
    )


# ============================================================================
# Configuration File Fixtures
# ============================================================================

@pytest.fixture
def sample_config_yaml(temp_dir):
    """Create sample YAML configuration file."""
    config_path = temp_dir / "test_config.yaml"
    config = {
        "model": {
            "base_model_name": "microsoft/Mistral-7B-v0.1",
            "quantization": False,
        },
        "lora": {
            "r": 16,
            "alpha": 16,
            "dropout": 0.05,
        },
        "training": {
            "learning_rate": 5e-5,
            "batch_size": 2,
            "num_epochs": 1,
        },
    }
    
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as requiring external API access"
    )


# ============================================================================
# Test Data Cleanup
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts(request):
    """Automatically cleanup test artifacts after each test."""
    yield
    # Cleanup logic here if needed
    pass