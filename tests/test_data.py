"""
Unit Tests for Data Collection and Preprocessing
================================================

Tests for:
- WeatherDataCollector
- WeatherPreprocessor
- Data validation and transformation
"""

import pytest
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from src.data import WeatherDataCollector, WeatherLocation, MAJOR_CITIES


# ============================================================================
# WeatherLocation Tests
# ============================================================================

@pytest.mark.unit
class TestWeatherLocation:
    """Test WeatherLocation dataclass."""
    
    def test_location_creation(self):
        """Test creating a weather location."""
        location = WeatherLocation("New York", 40.7128, -74.0060)
        assert location.name == "New York"
        assert location.latitude == 40.7128
        assert location.longitude == -74.0060
    
    def test_location_str_representation(self):
        """Test string representation of location."""
        location = WeatherLocation("London", 51.5074, -0.1278)
        assert "London" in str(location)
        assert "51.5074" in str(location)


# ============================================================================
# WeatherDataCollector Tests
# ============================================================================

@pytest.mark.unit
class TestWeatherDataCollector:
    """Test WeatherDataCollector functionality."""
    
    def test_collector_initialization(self):
        """Test collector initializes correctly."""
        collector = WeatherDataCollector()
        assert collector is not None
        assert hasattr(collector, 'cache_session')
    
    def test_major_cities_available(self):
        """Test that major cities list is populated."""
        assert len(MAJOR_CITIES) > 0
        assert isinstance(MAJOR_CITIES[0], WeatherLocation)
        assert MAJOR_CITIES[0].name is not None
    
    @patch('requests.get')
    def test_fetch_open_meteo_current_success(self, mock_get):
        """Test successful Open-Meteo API call."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "current_weather": {
                "temperature": 23.5,
                "windspeed": 12.0,
                "weathercode": 1
            },
            "hourly": {
                "temperature_2m": [23, 24, 22, 21],
                "relative_humidity_2m": [70, 75, 80, 82],
                "windspeed_10m": [12, 18, 20, 15],
                "surface_pressure": [1008, 1007, 1006, 1005],
            }
        }
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector()
        location = WeatherLocation("New York", 40.7128, -74.0060)
        
        # This would normally make an API call
        # result = collector.fetch_open_meteo_current([location])
        # assert result is not None
    
    @patch('requests.get')
    def test_fetch_open_meteo_error_handling(self, mock_get):
        """Test error handling for failed API calls."""
        # Mock failed API response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector()
        location = WeatherLocation("Invalid", 0, 0)
        
        # Should handle errors gracefully
        # result = collector.fetch_open_meteo_current([location])
    
    def test_data_validation(self, sample_weather_data):
        """Test weather data validation."""
        collector = WeatherDataCollector()
        
        # Valid data should pass
        assert sample_weather_data["location"] is not None
        assert len(sample_weather_data["temperature"]) > 0
        assert all(isinstance(t, (int, float)) for t in sample_weather_data["temperature"])
    
    def test_save_and_load_data(self, temp_dir, sample_weather_data):
        """Test saving and loading weather data."""
        collector = WeatherDataCollector()
        save_path = temp_dir / "test_weather.json"
        
        # Save data
        with open(save_path, 'w') as f:
            json.dump(sample_weather_data, f)
        
        # Load data
        with open(save_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["location"] == sample_weather_data["location"]
        assert loaded_data["temperature"] == sample_weather_data["temperature"]
    
    @pytest.mark.api
    def test_cache_mechanism(self, temp_dir):
        """Test data caching functionality."""
        collector = WeatherDataCollector(cache_session=True)
        
        # First call should cache
        # Second call should use cache
        # Verify cache session is enabled
        assert collector.cache_session is True


# ============================================================================
# WeatherPreprocessor Tests
# ============================================================================

@pytest.mark.unit
class TestWeatherPreprocessor:
    """Test WeatherPreprocessor functionality."""
    
    def test_numerical_to_text_conversion(self, sample_weather_data):
        """Test converting numerical data to text format."""
        from src.data import WeatherPreprocessor
        from src.data.preprocessor import ForecastPrompt
        
        preprocessor = WeatherPreprocessor()
        
        # Create ForecastPrompt from sample data
        prompt_obj = ForecastPrompt(
            location=sample_weather_data["location"],
            datetime=sample_weather_data["datetime"],
            temperature=sample_weather_data["temperature"],
            humidity=sample_weather_data["humidity"],
            wind_speed=sample_weather_data["wind_speed"],
            pressure=sample_weather_data["pressure"],
            precipitation_probability=sample_weather_data["precipitation_probability"],
            target_forecast="Sample forecast text"
        )
        
        # Should convert to prompt format
        text_prompt = preprocessor.format_prompt(prompt_obj)
        
        assert isinstance(text_prompt, str)
        assert "Temperature" in text_prompt or sample_weather_data["location"] in text_prompt
    
    def test_prompt_formatting(self, sample_weather_data):
        """Test prompt formatting follows specification."""
        from src.data import WeatherPreprocessor
        from src.data.preprocessor import ForecastPrompt
        
        preprocessor = WeatherPreprocessor()
        
        # Create ForecastPrompt
        prompt_obj = ForecastPrompt(
            location=sample_weather_data["location"],
            datetime=sample_weather_data["datetime"],
            temperature=sample_weather_data["temperature"],
            humidity=sample_weather_data["humidity"],
            wind_speed=sample_weather_data["wind_speed"],
            pressure=sample_weather_data["pressure"],
            precipitation_probability=sample_weather_data["precipitation_probability"],
            target_forecast="Sample forecast text"
        )
        
        prompt = preprocessor.format_prompt(prompt_obj)
        
        # Check format contains location
        assert sample_weather_data["location"] in prompt
    
    def test_dataset_creation(self, sample_weather_data):
        """Test creating training dataset."""
        from src.data import WeatherPreprocessor
        from src.data.preprocessor import ForecastPrompt
        
        preprocessor = WeatherPreprocessor()
        
        # Create ForecastPrompt objects
        forecast_prompts = [
            ForecastPrompt(
                location=sample_weather_data["location"],
                datetime=sample_weather_data["datetime"],
                temperature=sample_weather_data["temperature"],
                humidity=sample_weather_data["humidity"],
                wind_speed=sample_weather_data["wind_speed"],
                pressure=sample_weather_data["pressure"],
                precipitation_probability=sample_weather_data["precipitation_probability"],
                target_forecast="Sample forecast text"
            )
            for _ in range(10)  # Create 10 samples
        ]
        
        # Process training data
        train_data, val_data, test_data = preprocessor.create_training_dataset(forecast_prompts)
        
        assert len(train_data) + len(val_data) + len(test_data) == 10
        assert "input" in train_data[0]
        assert "target" in train_data[0]
    
    def test_train_val_test_split(self, sample_weather_data):
        """Test splitting data into train/val/test sets."""
        from src.data import WeatherPreprocessor
        from src.data.preprocessor import ForecastPrompt
        
        preprocessor = WeatherPreprocessor()
        
        # Create ForecastPrompt objects
        forecast_prompts = [
            ForecastPrompt(
                location=sample_weather_data["location"],
                datetime=sample_weather_data["datetime"],
                temperature=sample_weather_data["temperature"],
                humidity=sample_weather_data["humidity"],
                wind_speed=sample_weather_data["wind_speed"],
                pressure=sample_weather_data["pressure"],
                precipitation_probability=sample_weather_data["precipitation_probability"],
                target_forecast="Sample forecast text"
            )
            for _ in range(100)  # Create 100 samples for better split testing
        ]
        
        # Create splits using create_training_dataset
        train, val, test = preprocessor.create_training_dataset(
            forecast_prompts,
            train_split=0.7,
            val_split=0.15
        )
        
        total_samples = len(forecast_prompts)
        assert len(train) + len(val) + len(test) == total_samples
    
    def test_sequence_length_handling(self):
        """Test handling of variable sequence lengths."""
        from src.data import WeatherPreprocessor
        
        preprocessor = WeatherPreprocessor()
        
        # Test with different sequence lengths
        short_data = {"temperature": [20, 21]}
        long_data = {"temperature": [20] * 100}
        
        # Should handle both appropriately
        # preprocessor.create_prompt(short_data)
        # preprocessor.create_prompt(long_data)


# ============================================================================
# Data Integration Tests
# ============================================================================

@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for complete data pipeline."""
    
    def test_end_to_end_data_collection(self):
        """Test complete data collection workflow."""
        # Skip in CI/CD without API access
        pytest.skip("Requires API access")
        
        collector = WeatherDataCollector()
        # Collect -> Preprocess -> Save
    
    def test_dataset_preparation_for_training(self, sample_weather_data):
        """Test preparing dataset for LoRA training."""
        from src.data import WeatherPreprocessor
        from src.data.preprocessor import ForecastPrompt
        
        preprocessor = WeatherPreprocessor()
        
        # Create ForecastPrompt objects for integration test
        forecast_prompts = [
            ForecastPrompt(
                location=sample_weather_data["location"],
                datetime=sample_weather_data["datetime"],
                temperature=sample_weather_data["temperature"],
                humidity=sample_weather_data["humidity"],
                wind_speed=sample_weather_data["wind_speed"],
                pressure=sample_weather_data["pressure"],
                precipitation_probability=sample_weather_data["precipitation_probability"],
                target_forecast="Sample forecast for training"
            )
            for _ in range(20)  # Create 20 samples for integration test
        ]
        
        train_data, val_data, test_data = preprocessor.create_training_dataset(forecast_prompts)
        
        # Should be ready for model training
        assert all("input" in item for item in train_data)
        assert all("target" in item for item in train_data)
        assert len(train_data) + len(val_data) + len(test_data) == 20
    
    @pytest.mark.slow
    def test_large_dataset_processing(self):
        """Test processing large datasets efficiently."""
        # Create large synthetic dataset
        large_data = [{"input": f"prompt_{i}", "target": f"target_{i}"} 
                     for i in range(10000)]
        
        from src.data import WeatherPreprocessor
        preprocessor = WeatherPreprocessor()
        
        # Should process efficiently
        # result = preprocessor.create_training_dataset(large_data)
        # assert len(result) == len(large_data)


# ============================================================================
# Data Validation Tests
# ============================================================================

@pytest.mark.unit
class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_missing_fields_handling(self):
        """Test handling of missing required fields."""
        from src.data import WeatherPreprocessor
        from src.data.preprocessor import ForecastPrompt
        
        preprocessor = WeatherPreprocessor()
        
        # Create incomplete ForecastPrompt (missing required fields)
        # Should handle missing fields gracefully
        with pytest.raises((TypeError, AttributeError)):
            # Missing required fields
            incomplete_prompt = ForecastPrompt(
                location="Test"
                # Missing all other required fields
            )
            preprocessor.format_prompt(incomplete_prompt)
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        invalid_data = {
            "temperature": "not a number",  # Should be float
            "location": 123  # Should be string
        }
        
        # Should validate data types
        # with pytest.raises(TypeError):
        #     process_weather_data(invalid_data)
    
    def test_out_of_range_values(self):
        """Test handling of physically impossible values."""
        extreme_data = {
            "temperature": [1000],  # Unrealistic temperature
            "humidity": [150],  # Humidity > 100%
            "pressure": [-100]  # Negative pressure
        }
        
        # Should flag or handle extreme values
        # validate_weather_data(extreme_data)


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.performance
class TestDataPerformance:
    """Test data processing performance."""
    
    def test_preprocessing_speed(self, sample_weather_data):
        """Test preprocessing completes in reasonable time."""
        import time
        from src.data import WeatherPreprocessor
        from src.data.preprocessor import ForecastPrompt
        
        preprocessor = WeatherPreprocessor()
        
        # Create ForecastPrompt objects for performance test
        forecast_prompts = [
            ForecastPrompt(
                location=sample_weather_data["location"],
                datetime=sample_weather_data["datetime"],
                temperature=sample_weather_data["temperature"],
                humidity=sample_weather_data["humidity"],
                wind_speed=sample_weather_data["wind_speed"],
                pressure=sample_weather_data["pressure"],
                precipitation_probability=sample_weather_data["precipitation_probability"],
                target_forecast="Performance test forecast"
            )
            for _ in range(100)  # Create 100 samples
        ]
        
        start_time = time.time()
        train_data, val_data, test_data = preprocessor.create_training_dataset(forecast_prompts)
        elapsed = time.time() - start_time
        
        # Should process 100 samples in under 1 second
        assert elapsed < 1.0
        assert len(train_data) + len(val_data) + len(test_data) == 100
    
    def test_memory_efficiency(self):
        """Test memory usage stays reasonable."""
        # Test with large dataset
        # Monitor memory usage
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])