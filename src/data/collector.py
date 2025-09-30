"""
Data Collection Module for Weather Forecasting LoRA
===================================================

This module handles data collection from various weather APIs and sources:
- ERA5 reanalysis data from Copernicus Climate Data Store
- Open-Meteo API for current and forecast data
- NOAA Global Forecast System (GFS) data
- Historical weather observations

Following the project specification for numerical → text mapping.
"""

import requests
import pandas as pd
import numpy as np
import xarray as xr
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import openmeteo_requests
import requests_cache
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WeatherLocation:
    """Represents a weather location with coordinates and metadata."""
    name: str
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    timezone: Optional[str] = None
    country: Optional[str] = None


@dataclass
class WeatherObservation:
    """Represents a single weather observation with all variables."""
    location: WeatherLocation
    datetime: datetime
    temperature_c: float
    humidity_pct: float
    pressure_hpa: float
    wind_speed_kph: float
    wind_direction_deg: Optional[float] = None
    precipitation_mm: Optional[float] = None
    precipitation_probability: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    visibility_km: Optional[float] = None
    weather_code: Optional[int] = None


class WeatherDataCollector:
    """
    Main class for collecting weather data from multiple sources.
    
    Implements the data collection strategy outlined in the project:
    - Numerical weather variables (temp, humidity, pressure, wind)
    - Time series data for training LoRA adapters
    - Structured format for prompt templates
    """
    
    def __init__(self, cache_session: bool = True):
        """
        Initialize the weather data collector.
        
        Args:
            cache_session: Whether to use caching for API requests
        """
        self.cache_session = cache_session
        if cache_session:
            # Setup caching for API requests
            cache = requests_cache.CachedSession('.cache', expire_after=3600)
            self.openmeteo = openmeteo_requests.Client(session=cache)
        else:
            self.openmeteo = openmeteo_requests.Client()
            
        # API endpoints
        self.open_meteo_url = "https://api.open-meteo.com/v1/forecast"
        self.open_meteo_historical_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        
        logger.info("WeatherDataCollector initialized")
    
    def fetch_open_meteo_current(
        self,
        locations: List[WeatherLocation],
        days_ahead: int = 7
    ) -> List[WeatherObservation]:
        """
        Fetch current weather and forecasts from Open-Meteo API.
        
        Args:
            locations: List of WeatherLocation objects
            days_ahead: Number of days to forecast ahead
            
        Returns:
            List of WeatherObservation objects
        """
        logger.info(f"Fetching Open-Meteo data for {len(locations)} locations")
        
        observations = []
        
        for location in locations:
            try:
                # Prepare API parameters
                params = {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "current": [
                        "temperature_2m", 
                        "relative_humidity_2m",
                        "pressure_msl",
                        "wind_speed_10m",
                        "wind_direction_10m"
                    ],
                    "hourly": [
                        "temperature_2m",
                        "relative_humidity_2m", 
                        "pressure_msl",
                        "wind_speed_10m",
                        "wind_direction_10m",
                        "precipitation_probability",
                        "precipitation",
                        "cloud_cover"
                    ],
                    "forecast_days": days_ahead,
                    "timezone": "auto"
                }
                
                # Make API request
                responses = self.openmeteo.weather_api(self.open_meteo_url, params=params)
                response = responses[0]
                
                # Process current weather
                current = response.Current()
                current_obs = WeatherObservation(
                    location=location,
                    datetime=datetime.fromtimestamp(current.Time()),
                    temperature_c=current.Variables(0).Value(),
                    humidity_pct=current.Variables(1).Value(),
                    pressure_hpa=current.Variables(2).Value(),
                    wind_speed_kph=current.Variables(3).Value() * 3.6,  # m/s to km/h
                    wind_direction_deg=current.Variables(4).Value()
                )
                observations.append(current_obs)
                
                # Process hourly forecasts
                hourly = response.Hourly()
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s"),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    ),
                    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                    "humidity": hourly.Variables(1).ValuesAsNumpy(),
                    "pressure": hourly.Variables(2).ValuesAsNumpy(),
                    "wind_speed": hourly.Variables(3).ValuesAsNumpy(),
                    "wind_direction": hourly.Variables(4).ValuesAsNumpy(),
                    "precipitation_probability": hourly.Variables(5).ValuesAsNumpy(),
                    "precipitation": hourly.Variables(6).ValuesAsNumpy(),
                    "cloud_cover": hourly.Variables(7).ValuesAsNumpy()
                }
                
                # Convert to WeatherObservation objects
                for i in range(len(hourly_data["date"])):
                    obs = WeatherObservation(
                        location=location,
                        datetime=hourly_data["date"][i].to_pydatetime(),
                        temperature_c=hourly_data["temperature_2m"][i],
                        humidity_pct=hourly_data["humidity"][i],
                        pressure_hpa=hourly_data["pressure"][i],
                        wind_speed_kph=hourly_data["wind_speed"][i] * 3.6,
                        wind_direction_deg=hourly_data["wind_direction"][i],
                        precipitation_mm=hourly_data["precipitation"][i],
                        precipitation_probability=hourly_data["precipitation_probability"][i] / 100.0,
                        cloud_cover_pct=hourly_data["cloud_cover"][i]
                    )
                    observations.append(obs)
                
                logger.info(f"Collected {len(hourly_data['date']) + 1} observations for {location.name}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching data for {location.name}: {e}")
                continue
        
        logger.info(f"Total observations collected: {len(observations)}")
        return observations
    
    def fetch_open_meteo_historical(
        self,
        locations: List[WeatherLocation],
        start_date: str,
        end_date: str
    ) -> List[WeatherObservation]:
        """
        Fetch historical weather data from Open-Meteo.
        
        Args:
            locations: List of WeatherLocation objects
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of WeatherObservation objects
        """
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        
        observations = []
        
        for location in locations:
            try:
                params = {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "start_date": start_date,
                    "end_date": end_date,
                    "hourly": [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "pressure_msl", 
                        "wind_speed_10m",
                        "wind_direction_10m",
                        "precipitation",
                        "cloud_cover"
                    ],
                    "timezone": "UTC"
                }
                
                responses = self.openmeteo.weather_api(self.open_meteo_historical_url, params=params)
                response = responses[0]
                
                hourly = response.Hourly()
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s"),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    ),
                    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                    "humidity": hourly.Variables(1).ValuesAsNumpy(),
                    "pressure": hourly.Variables(2).ValuesAsNumpy(),
                    "wind_speed": hourly.Variables(3).ValuesAsNumpy(),
                    "wind_direction": hourly.Variables(4).ValuesAsNumpy(),
                    "precipitation": hourly.Variables(5).ValuesAsNumpy(),
                    "cloud_cover": hourly.Variables(6).ValuesAsNumpy()
                }
                
                # Convert to WeatherObservation objects
                for i in range(len(hourly_data["date"])):
                    obs = WeatherObservation(
                        location=location,
                        datetime=hourly_data["date"][i].to_pydatetime(),
                        temperature_c=hourly_data["temperature_2m"][i],
                        humidity_pct=hourly_data["humidity"][i],
                        pressure_hpa=hourly_data["pressure"][i],
                        wind_speed_kph=hourly_data["wind_speed"][i] * 3.6,
                        wind_direction_deg=hourly_data["wind_direction"][i],
                        precipitation_mm=hourly_data["precipitation"][i],
                        cloud_cover_pct=hourly_data["cloud_cover"][i]
                    )
                    observations.append(obs)
                
                logger.info(f"Collected {len(hourly_data['date'])} historical observations for {location.name}")
                
                # Rate limiting  
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching historical data for {location.name}: {e}")
                continue
        
        logger.info(f"Total historical observations collected: {len(observations)}")
        return observations
    
    def observations_to_dataframe(self, observations: List[WeatherObservation]) -> pd.DataFrame:
        """
        Convert list of WeatherObservation objects to pandas DataFrame.
        
        Args:
            observations: List of WeatherObservation objects
            
        Returns:
            pandas DataFrame with weather data
        """
        data = []
        for obs in observations:
            row = {
                'location_name': obs.location.name,
                'latitude': obs.location.latitude,
                'longitude': obs.location.longitude,
                'datetime': obs.datetime,
                'temperature_c': obs.temperature_c,
                'humidity_pct': obs.humidity_pct,
                'pressure_hpa': obs.pressure_hpa,
                'wind_speed_kph': obs.wind_speed_kph,
                'wind_direction_deg': obs.wind_direction_deg,
                'precipitation_mm': obs.precipitation_mm or 0.0,
                'precipitation_probability': obs.precipitation_probability or 0.0,
                'cloud_cover_pct': obs.cloud_cover_pct or 0.0,
                'visibility_km': obs.visibility_km,
                'weather_code': obs.weather_code
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.sort_values(['location_name', 'datetime'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def save_data(self, observations: List[WeatherObservation], filepath: str):
        """
        Save weather observations to file.
        
        Args:
            observations: List of WeatherObservation objects
            filepath: Path to save the data
        """
        df = self.observations_to_dataframe(observations)
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        elif filepath.endswith('.json'):
            df.to_json(filepath, orient='records', date_format='iso')
        else:
            raise ValueError("Unsupported file format. Use .csv, .parquet, or .json")
        
        logger.info(f"Saved {len(observations)} observations to {filepath}")


# Predefined major city locations for testing
MAJOR_CITIES = [
    WeatherLocation("New York", 40.7128, -74.0060, country="USA"),
    WeatherLocation("London", 51.5074, -0.1278, country="UK"),
    WeatherLocation("Tokyo", 35.6762, 139.6503, country="Japan"),
    WeatherLocation("Paris", 48.8566, 2.3522, country="France"),
    WeatherLocation("Sydney", -33.8688, 151.2093, country="Australia"),
    WeatherLocation("Berlin", 52.5200, 13.4050, country="Germany"),
    WeatherLocation("Toronto", 43.6532, -79.3832, country="Canada"),
    WeatherLocation("Mumbai", 19.0760, 72.8777, country="India"),
    WeatherLocation("São Paulo", -23.5505, -46.6333, country="Brazil"),
    WeatherLocation("Cairo", 30.0444, 31.2357, country="Egypt"),
]


if __name__ == "__main__":
    # Example usage
    collector = WeatherDataCollector()
    
    # Test with a few cities
    test_locations = MAJOR_CITIES[:3]
    
    # Fetch current weather
    current_observations = collector.fetch_open_meteo_current(test_locations, days_ahead=3)
    collector.save_data(current_observations, "data/current_weather.csv")
    
    # Fetch historical data (last 30 days)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    historical_observations = collector.fetch_open_meteo_historical(
        test_locations, start_date, end_date
    )
    collector.save_data(historical_observations, "data/historical_weather.csv")
    
    print(f"Collected {len(current_observations)} current observations")
    print(f"Collected {len(historical_observations)} historical observations")