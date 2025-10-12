"""
Tests Package for Weather Forecasting LoRA
==========================================

Test suite for comprehensive testing of the weather forecasting LoRA implementation.

Test Structure:
- test_data.py: Data collection and preprocessing tests
- test_models.py: LoRA model and training tests
- test_evaluation.py: Evaluation framework tests
- test_inference.py: Inference and deployment tests
- test_rl.py: PPO and RLHF tests

Usage:
    # Run all tests
    pytest tests/

    # Run specific test file
    pytest tests/test_data.py

    # Run with markers
    pytest tests/ -m unit
    pytest tests/ -m "not slow"

    # Run with coverage
    pytest tests/ --cov=src --cov-report=html
"""

__version__ = "1.0.0"
__author__ = "Weather Forecasting LoRA Team"

# Test markers documentation
MARKERS = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for workflows",
    "slow": "Slow-running tests (skip with '-m not slow')",
    "gpu": "Tests requiring GPU",
    "api": "Tests requiring external API access",
    "performance": "Performance and benchmark tests",
}

# Test configuration
TEST_CONFIG = {
    "timeout": 300,  # Test timeout in seconds
    "warnings": "ignore",  # Ignore warnings by default
    "verbose": True,
}

__all__ = [
    "MARKERS",
    "TEST_CONFIG",
]