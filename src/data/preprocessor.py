"""
Data Preprocessing Module for Weather Forecasting LoRA  
======================================================

This module handles data preprocessing and preparation for LoRA training:
- Convert numerical weather data to prompt templates
- Create training/validation datasets
- Generate target forecast text
- Handle data augmentation and balancing

Following the numerical → text mapping strategy from the project specification.
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ForecastPrompt:
    """Represents a structured prompt for weather forecasting."""
    location: str
    datetime: str
    temperature: List[float]
    humidity: List[float] 
    wind_speed: List[float]
    pressure: List[float]
    precipitation_probability: List[float]
    target_forecast: str


class WeatherPreprocessor:
    """
    Preprocesses weather data for LoRA training following project specifications.
    
    Implements the prompt template format:
    - Serialized numerical weather variables
    - Time series data (next 4-6 hours)
    - Natural language targets
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the weather preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Prompt templates following project specification
        self.prompt_template = """Weather data for {location} on {datetime}:
- Temperature (°C): {temperature}
- Humidity (%): {humidity}
- Wind speed (km/h): {wind_speed}
- Pressure (hPa): {pressure}
- Precipitation probability: {precipitation_probability}

Generate a forecast bulletin:"""

        # Weather description templates for generating realistic forecasts
        self.weather_templates = {
            'clear': [
                "Clear skies with {temp_desc} temperatures around {temp_range}°C.",
                "Sunny conditions expected with temperatures reaching {temp_range}°C.",
                "Bright and clear weather with {temp_desc} temperatures near {temp_range}°C."
            ],
            'cloudy': [
                "Partly cloudy skies with temperatures around {temp_range}°C.",
                "Overcast conditions with {temp_desc} temperatures near {temp_range}°C.",
                "Cloudy weather expected with temperatures reaching {temp_range}°C."
            ],
            'rainy': [
                "Showers expected with temperatures around {temp_range}°C. Precipitation chances {precip_desc}.",
                "Rain likely with {temp_desc} temperatures near {temp_range}°C. {precip_desc} chance of precipitation.",
                "Wet conditions forecast with temperatures reaching {temp_range}°C. {precip_desc} probability of rain."
            ],
            'windy': [
                "Breezy conditions with winds up to {wind_speed} km/h. Temperatures around {temp_range}°C.",
                "Windy weather expected with gusts reaching {wind_speed} km/h. {temp_desc} temperatures near {temp_range}°C.",
                "Strong winds up to {wind_speed} km/h forecast. Temperatures around {temp_range}°C."
            ]
        }
        
        logger.info("WeatherPreprocessor initialized")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'sequence_length': 4,  # Number of hours in sequence
            'temperature_bins': [-10, 0, 10, 20, 30, 40],
            'humidity_bins': [0, 30, 60, 80, 100],
            'wind_bins': [0, 10, 20, 40, 60],
            'pressure_bins': [980, 1000, 1020, 1040],
            'precip_bins': [0, 0.2, 0.5, 0.8, 1.0]
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults
            default_config.update(config)
        
        return default_config
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        sequence_length: int = 4,
        forecast_horizon: int = 6
    ) -> List[ForecastPrompt]:
        """
        Create training sequences from weather data.
        
        Args:
            df: DataFrame with weather observations
            sequence_length: Number of hours in input sequence
            forecast_horizon: Hours ahead to forecast
            
        Returns:
            List of ForecastPrompt objects
        """
        logger.info(f"Creating sequences with length {sequence_length}, horizon {forecast_horizon}")
        
        sequences = []
        
        # Group by location
        for location, location_data in df.groupby('location_name'):
            location_data = location_data.sort_values('datetime').reset_index(drop=True)
            
            # Create sequences for this location
            for i in range(len(location_data) - sequence_length - forecast_horizon):
                try:
                    # Input sequence (current + next few hours)
                    input_data = location_data.iloc[i:i+sequence_length]
                    
                    # Target data (future state for forecast)
                    target_data = location_data.iloc[i+sequence_length:i+sequence_length+forecast_horizon]
                    
                    # Extract sequences
                    temperature_seq = input_data['temperature_c'].tolist()
                    humidity_seq = input_data['humidity_pct'].tolist()
                    wind_seq = input_data['wind_speed_kph'].tolist()
                    pressure_seq = input_data['pressure_hpa'].tolist()
                    precip_seq = input_data['precipitation_probability'].tolist()
                    
                    # Generate target forecast text
                    target_forecast = self._generate_forecast_text(
                        input_data.iloc[-1],  # Last input observation
                        target_data  # Future observations for target
                    )
                    
                    # Create prompt
                    prompt = ForecastPrompt(
                        location=location,
                        datetime=input_data.iloc[0]['datetime'].strftime("%Y-%m-%d %H:%M UTC"),
                        temperature=temperature_seq,
                        humidity=humidity_seq,
                        wind_speed=wind_seq,
                        pressure=pressure_seq,
                        precipitation_probability=precip_seq,
                        target_forecast=target_forecast
                    )
                    
                    sequences.append(prompt)
                    
                except Exception as e:
                    logger.warning(f"Error creating sequence at index {i} for {location}: {e}")
                    continue
        
        logger.info(f"Created {len(sequences)} training sequences")
        return sequences
    
    def _generate_forecast_text(
        self, 
        current_obs: pd.Series, 
        future_obs: pd.DataFrame
    ) -> str:
        """
        Generate natural language forecast text from weather data.
        
        Args:
            current_obs: Current weather observation
            future_obs: Future weather observations
            
        Returns:
            Natural language forecast text
        """
        # Analyze future conditions
        avg_temp = future_obs['temperature_c'].mean()
        max_temp = future_obs['temperature_c'].max()
        min_temp = future_obs['temperature_c'].min()
        avg_humidity = future_obs['humidity_pct'].mean()
        max_wind = future_obs['wind_speed_kph'].max()
        avg_precip = future_obs['precipitation_probability'].mean()
        
        # Temperature descriptions
        if avg_temp < 0:
            temp_desc = "very cold"
        elif avg_temp < 10:
            temp_desc = "cold"
        elif avg_temp < 20:
            temp_desc = "cool"
        elif avg_temp < 30:
            temp_desc = "mild"
        else:
            temp_desc = "warm"
        
        # Temperature range
        temp_range = f"{min_temp:.0f}-{max_temp:.0f}"
        
        # Precipitation description
        if avg_precip < 0.2:
            precip_desc = "low"
        elif avg_precip < 0.5:
            precip_desc = "moderate"
        elif avg_precip < 0.8:
            precip_desc = "high"
        else:
            precip_desc = "very high"
        
        # Determine primary weather condition
        if avg_precip > 0.5:
            condition = 'rainy'
        elif max_wind > 30:
            condition = 'windy'
        elif avg_humidity > 80:
            condition = 'cloudy'
        else:
            condition = 'clear'
        
        # Select random template and format
        template = random.choice(self.weather_templates[condition])
        
        forecast_text = template.format(
            temp_desc=temp_desc,
            temp_range=temp_range,
            wind_speed=f"{max_wind:.0f}",
            precip_desc=precip_desc
        )
        
        # Add timing information
        if avg_precip > 0.3:
            timing_options = [
                "Showers expected by afternoon.",
                "Rain likely by evening.", 
                "Precipitation chances increase later.",
                "Wet conditions developing."
            ]
            forecast_text += " " + random.choice(timing_options)
        
        # Add wind information
        if max_wind > 20:
            wind_options = [
                f"Winds increasing to {max_wind:.0f} km/h.",
                f"Breezy conditions with gusts up to {max_wind:.0f} km/h.",
                f"Windy with speeds reaching {max_wind:.0f} km/h."
            ]
            forecast_text += " " + random.choice(wind_options)
        
        return forecast_text
    
    def format_prompt(self, forecast_prompt: ForecastPrompt) -> str:
        """
        Format a ForecastPrompt into the training prompt template.
        
        Args:
            forecast_prompt: ForecastPrompt object
            
        Returns:
            Formatted prompt string
        """
        # Format sequences as comma-separated strings
        temp_str = ", ".join([f"{t:.1f}" for t in forecast_prompt.temperature])
        humidity_str = ", ".join([f"{h:.0f}" for h in forecast_prompt.humidity])
        wind_str = ", ".join([f"{w:.1f}" for w in forecast_prompt.wind_speed])
        pressure_str = ", ".join([f"{p:.1f}" for p in forecast_prompt.pressure])
        precip_str = ", ".join([f"{p:.2f}" for p in forecast_prompt.precipitation_probability])
        
        return self.prompt_template.format(
            location=forecast_prompt.location,
            datetime=forecast_prompt.datetime,
            temperature=temp_str,
            humidity=humidity_str,
            wind_speed=wind_str,
            pressure=pressure_str,
            precipitation_probability=precip_str
        )
    
    def create_training_dataset(
        self,
        sequences: List[ForecastPrompt],
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create training, validation, and test datasets.
        
        Args:
            sequences: List of ForecastPrompt objects
            train_split: Fraction for training
            val_split: Fraction for validation (remainder goes to test)
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info(f"Creating datasets from {len(sequences)} sequences")
        
        # Shuffle sequences
        random.shuffle(sequences)
        
        # Calculate splits
        n_train = int(len(sequences) * train_split)
        n_val = int(len(sequences) * val_split)
        
        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:n_train + n_val]
        test_sequences = sequences[n_train + n_val:]
        
        # Convert to dataset format
        def convert_to_dataset(seqs: List[ForecastPrompt]) -> List[Dict]:
            dataset = []
            for seq in seqs:
                prompt = self.format_prompt(seq)
                item = {
                    'input': prompt,
                    'target': seq.target_forecast,
                    'prompt': prompt,
                    'response': seq.target_forecast,
                    'location': seq.location,
                    'datetime': seq.datetime
                }
                dataset.append(item)
            return dataset
        
        train_data = convert_to_dataset(train_sequences)
        val_data = convert_to_dataset(val_sequences)
        test_data = convert_to_dataset(test_sequences)
        
        logger.info(f"Created datasets - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_dataset(self, dataset: List[Dict], filepath: str):
        """
        Save dataset to file.
        
        Args:
            dataset: List of dataset items
            filepath: Path to save the dataset
        """
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2)
        elif filepath.endswith('.jsonl'):
            with open(filepath, 'w') as f:
                for item in dataset:
                    f.write(json.dumps(item) + '\n')
        else:
            raise ValueError("Unsupported format. Use .json or .jsonl")
        
        logger.info(f"Saved dataset with {len(dataset)} items to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        """
        Load dataset from file.
        
        Args:
            filepath: Path to dataset file
            
        Returns:
            List of dataset items
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                dataset = json.load(f)
        elif filepath.endswith('.jsonl'):
            dataset = []
            with open(filepath, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line.strip()))
        else:
            raise ValueError("Unsupported format. Use .json or .jsonl")
        
        logger.info(f"Loaded dataset with {len(dataset)} items from {filepath}")
        return dataset
    
    def get_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """
        Calculate statistics for a dataset.
        
        Args:
            dataset: List of dataset items
            
        Returns:
            Dictionary with statistics
        """
        if not dataset:
            return {}
        
        # Extract data for analysis
        locations = [item['location'] for item in dataset]
        responses = [item['response'] for item in dataset]
        
        # Calculate statistics
        stats = {
            'total_samples': len(dataset),
            'unique_locations': len(set(locations)),
            'locations': list(set(locations)),
            'avg_response_length': np.mean([len(r.split()) for r in responses]),
            'response_length_std': np.std([len(r.split()) for r in responses]),
            'min_response_length': min([len(r.split()) for r in responses]),
            'max_response_length': max([len(r.split()) for r in responses])
        }
        
        return stats


def main():
    """Example usage of the preprocessor."""
    # Initialize preprocessor
    preprocessor = WeatherPreprocessor()
    
    # Load weather data (assuming it exists)
    data_file = "data/historical_weather.csv"
    if Path(data_file).exists():
        df = pd.read_csv(data_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Create sequences
        sequences = preprocessor.create_sequences(df, sequence_length=4, forecast_horizon=6)
        
        # Create datasets
        train_data, val_data, test_data = preprocessor.create_training_dataset(sequences)
        
        # Save datasets
        Path("data/processed").mkdir(exist_ok=True)
        preprocessor.save_dataset(train_data, "data/processed/train.json")
        preprocessor.save_dataset(val_data, "data/processed/val.json")
        preprocessor.save_dataset(test_data, "data/processed/test.json")
        
        # Print statistics
        train_stats = preprocessor.get_dataset_statistics(train_data)
        print("Training dataset statistics:")
        for key, value in train_stats.items():
            print(f"  {key}: {value}")
        
    else:
        print(f"Data file {data_file} not found. Run data collection first.")


if __name__ == "__main__":
    main()