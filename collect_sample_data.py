#!/usr/bin/env python3
"""
Sample Data Collection Script
============================

This script demonstrates how to collect weather data for training the LoRA model.
Follows the project specification for gathering numerical weather data.

Run this script to:
1. Collect current weather data from multiple cities
2. Gather historical data for training
3. Preprocess data into training format
4. Save datasets for model training

Usage:
    python collect_sample_data.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import WeatherDataCollector, WeatherPreprocessor, MAJOR_CITIES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main data collection workflow."""
    logger.info("Starting weather data collection for LoRA training")
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)
    
    # Initialize collector
    collector = WeatherDataCollector(cache_session=True)
    
    # Select cities for training (use subset for demo)
    training_cities = MAJOR_CITIES[:5]  # New York, London, Tokyo, Paris, Sydney
    logger.info(f"Collecting data for {len(training_cities)} cities: {[city.name for city in training_cities]}")
    
    # === STEP 1: Collect Current Weather ===
    logger.info("Step 1: Collecting current weather and forecasts...")
    try:
        current_observations = collector.fetch_open_meteo_current(
            locations=training_cities,
            days_ahead=7  # Next 7 days
        )
        
        if current_observations:
            collector.save_data(current_observations, "data/raw/current_weather.csv")
            logger.info(f"✓ Saved {len(current_observations)} current observations")
        else:
            logger.warning("No current observations collected")
            
    except Exception as e:
        logger.error(f"Error collecting current weather: {e}")
    
    # === STEP 2: Collect Historical Weather ===
    logger.info("Step 2: Collecting historical weather data...")
    try:
        # Get last 60 days for training data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        historical_observations = collector.fetch_open_meteo_historical(
            locations=training_cities,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if historical_observations:
            collector.save_data(historical_observations, "data/raw/historical_weather.csv")
            logger.info(f"✓ Saved {len(historical_observations)} historical observations")
        else:
            logger.warning("No historical observations collected")
            
    except Exception as e:
        logger.error(f"Error collecting historical weather: {e}")
    
    # === STEP 3: Preprocess Data ===
    logger.info("Step 3: Preprocessing data for LoRA training...")
    try:
        preprocessor = WeatherPreprocessor()
        
        # Load historical data for processing
        if Path("data/raw/historical_weather.csv").exists():
            df = pd.read_csv("data/raw/historical_weather.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            logger.info(f"Loaded {len(df)} historical observations")
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            logger.info(f"Locations: {df['location_name'].unique().tolist()}")
            
            # Create training sequences
            sequences = preprocessor.create_sequences(
                df, 
                sequence_length=4,     # 4-hour input sequences  
                forecast_horizon=6     # Predict 6 hours ahead
            )
            
            if sequences:
                logger.info(f"✓ Created {len(sequences)} training sequences")
                
                # Split into train/val/test
                train_data, val_data, test_data = preprocessor.create_training_dataset(sequences)
                
                # Save datasets
                preprocessor.save_dataset(train_data, "data/processed/train.json")
                preprocessor.save_dataset(val_data, "data/processed/val.json") 
                preprocessor.save_dataset(test_data, "data/processed/test.json")
                
                # Print dataset statistics
                train_stats = preprocessor.get_dataset_statistics(train_data)
                logger.info("Training dataset statistics:")
                for key, value in train_stats.items():
                    logger.info(f"  {key}: {value}")
                
                # Show example
                logger.info("\n" + "="*50)
                logger.info("EXAMPLE TRAINING SAMPLE:")
                logger.info("="*50)
                example = train_data[0]
                logger.info("INPUT PROMPT:")
                logger.info(example['input'])
                logger.info("\nTARGET FORECAST:")
                logger.info(example['target'])
                logger.info("="*50)
                
            else:
                logger.warning("No sequences created from historical data")
        else:
            logger.warning("Historical weather data not found")
            
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
    
    # === STEP 4: Data Summary ===
    logger.info("Step 4: Data collection summary")
    
    files_created = []
    for filepath in [
        "data/raw/current_weather.csv",
        "data/raw/historical_weather.csv", 
        "data/processed/train.json",
        "data/processed/val.json",
        "data/processed/test.json"
    ]:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size / 1024  # KB
            files_created.append(f"  ✓ {filepath} ({size:.1f} KB)")
    
    logger.info("Files created:")
    for file_info in files_created:
        logger.info(file_info)
    
    if Path("data/processed/train.json").exists():
        logger.info("\n🎉 Data collection completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. Review the training data in data/processed/")
        logger.info("  2. Run the LoRA training script")
        logger.info("  3. Evaluate model performance")
    else:
        logger.warning("\n⚠️  Data collection incomplete. Check logs for errors.")


if __name__ == "__main__":
    main()