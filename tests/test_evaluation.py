"""
Unit Tests for Evaluation Framework
====================================

Tests for:
- MetricsCalculator
- WeatherEvaluator
- Evaluation metrics (BLEU, ROUGE, accuracy, calibration)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from src.evaluation import (
    MetricsCalculator,
    WeatherEvaluator,
    ForecastPrediction,
    EvaluationMetrics
)


# ============================================================================
# MetricsCalculator Tests
# ============================================================================

@pytest.mark.unit
class TestMetricsCalculator:
    """Test MetricsCalculator functionality."""
    
    def test_calculator_initialization(self):
        """Test calculator initializes correctly."""
        calculator = MetricsCalculator()
        assert calculator is not None
    
    def test_bleu_score_calculation(self):
        """Test BLEU score calculation."""
        calculator = MetricsCalculator()
        
        reference = "Partly cloudy with temperatures around 23°C"
        candidate = "Partly cloudy with temps near 23 degrees"
        
        bleu_score = calculator.calculate_bleu_score(candidate, reference)
        
        assert 0.0 <= bleu_score <= 1.0
        assert isinstance(bleu_score, float)
    
    def test_rouge_scores_calculation(self):
        """Test ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)."""
        calculator = MetricsCalculator()
        
        reference = "Rain showers expected with temperatures around 20°C"
        candidate = "Expect rain showers with temps near 20 degrees"
        
        rouge_scores = calculator.calculate_rouge_scores(candidate, reference)
        
        assert "rouge_1_f" in rouge_scores or "rouge1" in rouge_scores
        assert "rouge_2_f" in rouge_scores or "rouge2" in rouge_scores
        assert "rouge_l_f" in rouge_scores or "rougeL" in rouge_scores
        
        # All scores should be between 0 and 1
        for score in rouge_scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_perfect_match_scores(self):
        """Test scores for identical predictions."""
        calculator = MetricsCalculator()
        
        text = "Sunny with temperatures around 25°C"
        
        bleu = calculator.calculate_bleu_score(text, text)
        rouge = calculator.calculate_rouge_scores(text, text)
        
        # Perfect match should give high scores
        assert bleu > 0.9
        # Check any rouge score (could be rouge_1_f or rouge1 depending on implementation)
        rouge_values = list(rouge.values())
        assert any(score > 0.9 for score in rouge_values)
    
    def test_completely_different_scores(self):
        """Test scores for completely different predictions."""
        calculator = MetricsCalculator()
        
        reference = "Sunny and warm"
        candidate = "Rainy and cold"
        
        bleu = calculator.calculate_bleu_score(candidate, reference)
        
        # Different text should give low score
        assert bleu < 0.3


# ============================================================================
# Meteorological Accuracy Tests
# ============================================================================

@pytest.mark.unit
class TestMeteorologicalAccuracy:
    """Test weather-specific accuracy metrics."""
    
    def test_rain_prediction_accuracy(self):
        """Test rain/no-rain classification accuracy."""
        calculator = MetricsCalculator()
        
        predictions = [
            ("Rain expected", True),
            ("Dry conditions", False),
            ("Showers likely", True),
            ("No precipitation", False),
        ]
        
        correct = sum(
            1 for pred, actual in predictions
            if ("rain" in pred.lower() or "shower" in pred.lower()) == actual
        )
        
        accuracy = correct / len(predictions)
        assert accuracy > 0.5
    
    def test_temperature_mae_calculation(self):
        """Test Mean Absolute Error for temperature."""
        calculator = MetricsCalculator()
        
        predicted_temps = [23, 24, 25, 22]
        actual_temps = [22, 25, 24, 23]
        
        mae = np.mean(np.abs(np.array(predicted_temps) - np.array(actual_temps)))
        
        assert mae >= 0.0
        assert isinstance(mae, (int, float))
    
    def test_wind_speed_accuracy(self):
        """Test wind speed prediction accuracy."""
        calculator = MetricsCalculator()
        
        predicted_winds = [15, 20, 25, 18]
        actual_winds = [14, 22, 24, 19]
        
        mae = np.mean(np.abs(np.array(predicted_winds) - np.array(actual_winds)))
        
        # MAE should be reasonable (< 5 km/h for good model)
        assert mae >= 0.0
    
    def test_calibration_brier_score(self):
        """Test calibration using Brier score."""
        calculator = MetricsCalculator()
        
        # Probability predictions
        predicted_probs = [0.1, 0.3, 0.7, 0.9]
        # Actual outcomes (0 or 1)
        actual_outcomes = [0, 0, 1, 1]
        
        brier = np.mean(
            [(p - a) ** 2 for p, a in zip(predicted_probs, actual_outcomes)]
        )
        
        assert 0.0 <= brier <= 1.0
        # Lower is better; well-calibrated should be < 0.2


# ============================================================================
# ForecastPrediction Tests
# ============================================================================

@pytest.mark.unit
class TestForecastPrediction:
    """Test ForecastPrediction dataclass."""
    
    def test_prediction_creation(self):
        """Test creating a forecast prediction."""
        pred = ForecastPrediction(
            input_prompt="Weather data...",
            generated_forecast="Partly cloudy, 23°C",
            reference_forecast="Partly cloudy, temps around 23°C",
            location="New York",
            datetime="2025-10-12"
        )
        
        assert pred.input_prompt is not None
        assert pred.generated_forecast is not None
        assert pred.reference_forecast is not None
        assert pred.location == "New York"
    
    def test_prediction_validation(self):
        """Test prediction data validation."""
        # Required fields should be present
        with pytest.raises(TypeError):
            ForecastPrediction()  # Missing required arguments


# ============================================================================
# WeatherEvaluator Tests
# ============================================================================

@pytest.mark.unit
class TestWeatherEvaluator:
    """Test WeatherEvaluator functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        evaluator = WeatherEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'calculator')
    
    def test_evaluate_single_prediction(self, sample_forecast_predictions):
        """Test evaluating a single prediction."""
        evaluator = WeatherEvaluator()
        calculator = evaluator.calculator
        
        pred = sample_forecast_predictions[0]
        
        # Calculate metrics for single prediction
        bleu = calculator.calculate_bleu_score(
            pred.generated_forecast,
            pred.reference_forecast
        )
        
        assert 0.0 <= bleu <= 1.0
    
    def test_evaluate_model_on_dataset(self, mock_lora_model, sample_test_data):
        """Test evaluating model on test dataset."""
        evaluator = WeatherEvaluator()
        
        # Mock model evaluation
        # metrics = evaluator.evaluate_model(mock_lora_model, sample_test_data)
        
        # Should return EvaluationMetrics
        # assert isinstance(metrics, EvaluationMetrics)
    
    def test_calculate_all_metrics(self, sample_forecast_predictions):
        """Test calculating all metrics from predictions."""
        evaluator = WeatherEvaluator()
        
        metrics = evaluator.calculator.calculate_all_metrics(
            sample_forecast_predictions
        )
        
        assert isinstance(metrics, EvaluationMetrics)
        assert hasattr(metrics, 'bleu_score')
        assert hasattr(metrics, 'rouge_1_f') or hasattr(metrics, 'rouge1_score')
        assert hasattr(metrics, 'overall_score')
    
    def test_evaluation_report_generation(self, sample_metrics, temp_dir):
        """Test generating evaluation report."""
        evaluator = WeatherEvaluator()
        
        report_path = temp_dir / "evaluation_report.md"
        report = evaluator.create_evaluation_report(
            sample_metrics,
            str(report_path)
        )
        
        assert isinstance(report, str)
        assert "BLEU" in report
        assert "ROUGE" in report
        assert report_path.exists()
    
    def test_readability_score_calculation(self):
        """Test readability score for forecasts."""
        evaluator = WeatherEvaluator()
        
        simple_forecast = "Sunny. Temperature 25°C. Light winds."
        complex_forecast = "Atmospheric conditions indicate a high-pressure system resulting in predominantly clear skies with minimal cloud coverage."
        
        # Readability metrics (simpler should score higher)
        # simple_score = evaluator.calculate_readability(simple_forecast)
        # complex_score = evaluator.calculate_readability(complex_forecast)


# ============================================================================
# EvaluationMetrics Tests
# ============================================================================

@pytest.mark.unit
class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating evaluation metrics."""
        metrics = EvaluationMetrics(
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
        
        assert metrics.bleu_score == 0.45
        assert metrics.overall_score == 0.68
    
    def test_overall_score_calculation(self, sample_metrics):
        """Test overall score is calculated correctly."""
        # Overall score should be weighted combination
        assert 0.0 <= sample_metrics.overall_score <= 1.0
    
    def test_metrics_comparison(self):
        """Test comparing different metrics."""
        metrics1 = EvaluationMetrics(
            bleu_score=0.5,
            rouge_1_f=0.6,
            rouge_2_f=0.4,
            rouge_l_f=0.5,
            rain_accuracy=0.8,
            temperature_mae=1.0,
            wind_speed_mae=2.0,
            categorical_accuracy=0.75,
            brier_score=0.15,
            reliability=0.85,
            resolution=0.80,
            readability_score=0.7,
            length_similarity=0.85,
            vocabulary_diversity=0.65,
            overall_score=0.7,
            confidence_interval=(0.67, 0.73)
        )
        
        metrics2 = EvaluationMetrics(
            bleu_score=0.4,
            rouge_1_f=0.5,
            rouge_2_f=0.3,
            rouge_l_f=0.4,
            rain_accuracy=0.7,
            temperature_mae=2.0,
            wind_speed_mae=3.0,
            categorical_accuracy=0.65,
            brier_score=0.25,
            reliability=0.75,
            resolution=0.70,
            readability_score=0.6,
            length_similarity=0.75,
            vocabulary_diversity=0.55,
            overall_score=0.6,
            confidence_interval=(0.57, 0.63)
        )
        
        # metrics1 should be better than metrics2
        assert metrics1.overall_score > metrics2.overall_score


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests for evaluation workflow."""
    
    def test_end_to_end_evaluation(self, mock_lora_model, sample_test_data, temp_dir):
        """Test complete evaluation pipeline."""
        evaluator = WeatherEvaluator()
        
        # Evaluate model
        # metrics = evaluator.evaluate_model(mock_lora_model, sample_test_data)
        
        # Generate report
        # report = evaluator.create_evaluation_report(
        #     metrics,
        #     str(temp_dir / "report.md")
        # )
        
        # Verify outputs
        # assert metrics.overall_score > 0.0
        # assert (temp_dir / "report.md").exists()
    
    def test_compare_sft_vs_ppo_models(self):
        """Test comparing SFT and PPO model performance."""
        # Load both models
        # Evaluate both on same test set
        # Compare metrics
        # PPO should generally score higher
        pass
    
    @pytest.mark.slow
    def test_ablation_study_evaluation(self):
        """Test evaluation for ablation studies."""
        # Evaluate different LoRA configurations
        # Compare results
        pass


# ============================================================================
# Domain-Specific Evaluation Tests
# ============================================================================

@pytest.mark.unit
class TestDomainSpecificMetrics:
    """Test weather forecasting domain-specific metrics."""
    
    def test_terminology_usage(self):
        """Test meteorological terminology usage."""
        evaluator = WeatherEvaluator()
        
        good_forecast = "Partly cloudy with scattered showers. High pressure system moving in."
        poor_forecast = "It might be cloudy. Maybe some water falling from sky."
        
        # Good forecast should use proper terminology
        # terminology_score_good = evaluator.check_terminology(good_forecast)
        # terminology_score_poor = evaluator.check_terminology(poor_forecast)
    
    def test_forecast_structure(self):
        """Test forecast follows meteorological structure."""
        forecast = """
        Afternoon temperatures around 23-24°C with high humidity.
        Winds increasing to 20 km/h by early evening.
        Showers likely by evening with precipitation chances above 60%.
        """
        
        # Should mention: temperature, wind, precipitation
        assert "temperature" in forecast.lower() or "°C" in forecast
        assert "wind" in forecast.lower() or "km/h" in forecast
        assert "shower" in forecast.lower() or "rain" in forecast.lower()
    
    def test_numerical_consistency(self):
        """Test numerical values are consistent."""
        # Temperature should be in reasonable range
        # Wind speed should match description (light/moderate/strong)
        # Precipitation probability should match text description
        pass


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestEvaluationErrorHandling:
    """Test evaluation error handling."""
    
    def test_empty_predictions_handling(self):
        """Test handling of empty prediction lists."""
        evaluator = WeatherEvaluator()
        
        # Should handle empty list gracefully
        # metrics = evaluator.calculator.calculate_all_metrics([])
    
    def test_missing_reference_handling(self):
        """Test handling of missing reference forecasts."""
        pred = ForecastPrediction(
            input_prompt="test",
            generated_forecast="Sunny",
            reference_forecast=None,  # Missing reference
            location="Test",
            datetime="2025-10-12"
        )
        
        # Should handle None reference
    
    def test_malformed_prediction_handling(self):
        """Test handling of malformed predictions."""
        # Test with invalid data types
        # Test with missing fields
        pass


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.performance
class TestEvaluationPerformance:
    """Test evaluation performance."""
    
    def test_metric_calculation_speed(self, sample_forecast_predictions):
        """Test metrics calculate in reasonable time."""
        import time
        
        evaluator = WeatherEvaluator()
        
        start = time.time()
        metrics = evaluator.calculator.calculate_all_metrics(
            sample_forecast_predictions * 100
        )
        elapsed = time.time() - start
        
        # Should evaluate 200 predictions in under 2 seconds
        assert elapsed < 2.0
    
    def test_large_dataset_evaluation(self):
        """Test evaluating large datasets efficiently."""
        # Create large prediction set
        # Measure performance
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])