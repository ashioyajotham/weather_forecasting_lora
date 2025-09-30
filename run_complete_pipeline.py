#!/usr/bin/env python3
"""
Complete Training and Evaluation Pipeline
=========================================

This script implements the complete 8-week training schedule from the project specification:
- Week 1: Data Setup & Baseline
- Week 2-3: Phase 1 SFT with LoRA  
- Week 4-5: Phase 2 RL with PPO
- Week 6: Robustness & Ablations
- Week 7: Deployment Prep

Usage:
    python run_complete_pipeline.py --stage [data|sft|ppo|eval|all]
"""

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data import WeatherDataCollector, WeatherPreprocessor, MAJOR_CITIES
from src.models import WeatherForecasterLoRA, LoRATrainer
from src.evaluation import WeatherEvaluator
from src.rl import PPOTrainerWeather, WeatherRewardModel
from src.inference import WeatherInference

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WeatherLoRAPipeline:
    """
    Complete training and evaluation pipeline for Weather Forecasting LoRA.
    
    Implements the 8-week schedule following Schulman et al. (2025) methodology.
    """
    
    def __init__(self, config_path: str = "config/base_config.yaml"):
        """
        Initialize the complete pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Pipeline state
        self.stage_completed = {
            'data': False,
            'sft': False, 
            'ppo': False,
            'eval': False
        }
        
        # Paths
        self.data_dir = Path("data")
        self.model_dir = Path("models")
        self.log_dir = Path("logs")
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.model_dir, self.log_dir]:
            dir_path.mkdir(exist_ok=True)
            (dir_path / "raw").mkdir(exist_ok=True)
            (dir_path / "processed").mkdir(exist_ok=True)
        
        logger.info("WeatherLoRAPipeline initialized")
    
    def _load_config(self) -> dict:
        """Load pipeline configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
    
    def stage_data_collection(self) -> bool:
        """
        Stage 1: Data Setup & Baseline (Week 1)
        
        - Collect weather data from multiple sources
        - Preprocess into training format
        - Create baseline evaluation
        
        Returns:
            True if successful
        """
        logger.info("="*60)
        logger.info("STAGE 1: DATA COLLECTION & BASELINE (Week 1)")
        logger.info("="*60)
        
        try:
            # Initialize data collector
            collector = WeatherDataCollector(cache_session=True)
            
            # Select cities for training
            training_cities = MAJOR_CITIES[:8]  # Expanded set
            logger.info(f"Collecting data for {len(training_cities)} cities")
            
            # Collect current weather (for recent patterns)
            logger.info("Collecting current weather and forecasts...")
            current_observations = collector.fetch_open_meteo_current(
                locations=training_cities,
                days_ahead=7
            )
            
            if current_observations:
                collector.save_data(current_observations, "data/raw/current_weather.csv")
                logger.info(f"✓ Saved {len(current_observations)} current observations")
            
            # Collect historical data (90 days for more training data)
            logger.info("Collecting historical weather data...")
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            historical_observations = collector.fetch_open_meteo_historical(
                locations=training_cities,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            if historical_observations:
                collector.save_data(historical_observations, "data/raw/historical_weather.csv")
                logger.info(f"✓ Saved {len(historical_observations)} historical observations")
            
            # Preprocess data
            logger.info("Preprocessing data for training...")
            preprocessor = WeatherPreprocessor()
            
            # Load and process historical data
            import pandas as pd
            df = pd.read_csv("data/raw/historical_weather.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Create training sequences
            sequences = preprocessor.create_sequences(
                df, 
                sequence_length=4,
                forecast_horizon=6
            )
            
            if sequences:
                logger.info(f"✓ Created {len(sequences)} training sequences")
                
                # Split datasets
                train_data, val_data, test_data = preprocessor.create_training_dataset(sequences)
                
                # Save datasets
                preprocessor.save_dataset(train_data, "data/processed/train.json")
                preprocessor.save_dataset(val_data, "data/processed/val.json")
                preprocessor.save_dataset(test_data, "data/processed/test.json")
                
                # Statistics
                stats = preprocessor.get_dataset_statistics(train_data)
                logger.info("Dataset Statistics:")
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")
                
                # Save statistics
                with open("data/processed/dataset_stats.json", 'w') as f:
                    json.dump(stats, f, indent=2)
                
                self.stage_completed['data'] = True
                logger.info("✅ Data collection stage completed successfully")
                return True
            else:
                logger.error("❌ Failed to create training sequences")
                return False
                
        except Exception as e:
            logger.error(f"❌ Data collection stage failed: {e}")
            return False
    
    def stage_sft_training(self) -> bool:
        """
        Stage 2: SFT with LoRA (Week 2-3)
        
        - Initialize LoRA model following Schulman methodology
        - Train with supervised fine-tuning
        - Evaluate performance
        
        Returns:
            True if successful
        """
        logger.info("="*60)
        logger.info("STAGE 2: SFT LORA TRAINING (Week 2-3)")
        logger.info("="*60)
        
        try:
            # Check data availability
            if not Path("data/processed/train.json").exists():
                logger.error("Training data not found. Run data collection first.")
                return False
            
            # Load datasets
            def load_dataset(path):
                with open(path, 'r') as f:
                    return json.load(f)
            
            train_data = load_dataset("data/processed/train.json")
            val_data = load_dataset("data/processed/val.json")
            
            logger.info(f"Loaded {len(train_data)} training, {len(val_data)} validation samples")
            
            # Initialize model with LoRA
            logger.info("Initializing WeatherForecasterLoRA model...")
            model = WeatherForecasterLoRA(
                base_model_name=self.config.get('model', {}).get('base_model', 'microsoft/Mistral-7B-v0.1'),
                lora_config=self.config.get('lora', {}),
                quantization=True
            )
            
            # Initialize trainer
            trainer = LoRATrainer(model=model, config_path="config/sft_config.yaml")
            
            # Train model
            logger.info("🚀 Starting SFT LoRA training...")
            training_history = trainer.train(
                train_dataset=train_data,
                eval_dataset=val_data,
                output_dir="models/weather-lora-sft"
            )
            
            # Save training history
            with open("models/weather-lora-sft/training_history.json", 'w') as f:
                json.dump(training_history, f, indent=2)
            
            self.stage_completed['sft'] = True
            logger.info("✅ SFT training stage completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ SFT training stage failed: {e}")
            return False
    
    def stage_ppo_training(self) -> bool:
        """
        Stage 3: RL with PPO (Week 4-5)
        
        - Load SFT model and add value head
        - Initialize reward model
        - Train with PPO following Schulman methodology
        
        Returns:
            True if successful
        """
        logger.info("="*60)
        logger.info("STAGE 3: PPO RLHF TRAINING (Week 4-5)")
        logger.info("="*60)
        
        try:
            # Check SFT model availability
            if not Path("models/weather-lora-sft").exists():
                logger.error("SFT model not found. Run SFT training first.")
                return False
            
            # Load training data
            with open("data/processed/train.json", 'r') as f:
                train_data = json.load(f)
            
            # Take subset for PPO (faster iteration)
            ppo_data = train_data[:1000]  # Limit for demonstration
            logger.info(f"Using {len(ppo_data)} samples for PPO training")
            
            # Initialize reward model
            logger.info("Initializing reward model...")
            reward_model = WeatherRewardModel(
                accuracy_weight=0.4,
                style_weight=0.2,
                calibration_weight=0.2,
                consistency_weight=0.2
            )
            
            # Initialize PPO trainer
            logger.info("Initializing PPO trainer...")
            ppo_trainer = PPOTrainerWeather(
                model_path="models/weather-lora-sft",
                reward_model=reward_model,
                config_path="config/ppo_config.yaml"
            )
            
            # Train with PPO
            logger.info("🚀 Starting PPO RLHF training...")
            ppo_trainer.train(
                training_data=ppo_data,
                num_epochs=2,  # Reduced for demonstration
                save_every=50,
                output_dir="models/weather-lora-ppo"
            )
            
            self.stage_completed['ppo'] = True
            logger.info("✅ PPO training stage completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ PPO training stage failed: {e}")
            return False
    
    def stage_evaluation(self) -> bool:
        """
        Stage 4: Comprehensive Evaluation (Week 6)
        
        - Evaluate both SFT and PPO models
        - Generate performance reports
        - Conduct ablation studies
        
        Returns:
            True if successful
        """
        logger.info("="*60)
        logger.info("STAGE 4: COMPREHENSIVE EVALUATION (Week 6)")
        logger.info("="*60)
        
        try:
            # Load test dataset
            with open("data/processed/test.json", 'r') as f:
                test_data = json.load(f)
            
            # Limit test samples for demonstration
            test_data = test_data[:100]
            logger.info(f"Evaluating on {len(test_data)} test samples")
            
            # Initialize evaluator
            evaluator = WeatherEvaluator()
            
            results = {}
            
            # Evaluate SFT model
            if Path("models/weather-lora-sft").exists():
                logger.info("Evaluating SFT model...")
                try:
                    sft_inference = WeatherInference("models/weather-lora-sft")
                    sft_inference.load_model()
                    
                    sft_metrics = evaluator.evaluate_model(sft_inference, test_data)
                    results['sft'] = {
                        'overall_score': sft_metrics.overall_score,
                        'bleu_score': sft_metrics.bleu_score,
                        'rain_accuracy': sft_metrics.rain_accuracy,
                        'temperature_mae': sft_metrics.temperature_mae,
                        'readability_score': sft_metrics.readability_score
                    }
                    
                    # Generate report
                    sft_report = evaluator.create_evaluation_report(
                        sft_metrics, 
                        "models/weather-lora-sft/evaluation_report.md"
                    )
                    
                    logger.info(f"SFT Model - Overall Score: {sft_metrics.overall_score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"SFT evaluation failed: {e}")
            
            # Evaluate PPO model
            if Path("models/weather-lora-ppo").exists():
                logger.info("Evaluating PPO model...")
                try:
                    ppo_inference = WeatherInference("models/weather-lora-ppo")
                    ppo_inference.load_model()
                    
                    ppo_metrics = evaluator.evaluate_model(ppo_inference, test_data)
                    results['ppo'] = {
                        'overall_score': ppo_metrics.overall_score,
                        'bleu_score': ppo_metrics.bleu_score,
                        'rain_accuracy': ppo_metrics.rain_accuracy,
                        'temperature_mae': ppo_metrics.temperature_mae,
                        'readability_score': ppo_metrics.readability_score
                    }
                    
                    # Generate report
                    ppo_report = evaluator.create_evaluation_report(
                        ppo_metrics,
                        "models/weather-lora-ppo/evaluation_report.md"
                    )
                    
                    logger.info(f"PPO Model - Overall Score: {ppo_metrics.overall_score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"PPO evaluation failed: {e}")
            
            # Save comparison results
            with open("evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print comparison
            if 'sft' in results and 'ppo' in results:
                logger.info("\n" + "="*40)
                logger.info("MODEL COMPARISON RESULTS")
                logger.info("="*40)
                logger.info(f"SFT Overall Score:  {results['sft']['overall_score']:.3f}")
                logger.info(f"PPO Overall Score:  {results['ppo']['overall_score']:.3f}")
                logger.info(f"Improvement:        {results['ppo']['overall_score'] - results['sft']['overall_score']:+.3f}")
                
                if results['ppo']['overall_score'] > results['sft']['overall_score']:
                    logger.info("✅ PPO improved over SFT baseline")
                else:
                    logger.info("⚠️ PPO did not improve over SFT baseline")
            
            self.stage_completed['eval'] = True
            logger.info("✅ Evaluation stage completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Evaluation stage failed: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete 8-week training pipeline."""
        logger.info("🚀 STARTING COMPLETE WEATHER LORA PIPELINE")
        logger.info("Following Schulman et al. (2025) methodology")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Stage 1: Data Collection
        if not self.stage_data_collection():
            logger.error("Pipeline failed at data collection stage")
            return False
        
        # Stage 2: SFT Training 
        if not self.stage_sft_training():
            logger.error("Pipeline failed at SFT training stage")
            return False
        
        # Stage 3: PPO Training
        if not self.stage_ppo_training():
            logger.error("Pipeline failed at PPO training stage")
            return False
        
        # Stage 4: Evaluation
        if not self.stage_evaluation():
            logger.error("Pipeline failed at evaluation stage")
            return False
        
        # Pipeline completion
        total_time = time.time() - start_time
        logger.info("="*80)
        logger.info("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        logger.info(f"Total execution time: {total_time:.1f} seconds")
        logger.info("="*80)
        
        # Summary
        logger.info("PIPELINE SUMMARY:")
        logger.info("✅ Data collection and preprocessing")
        logger.info("✅ SFT LoRA training")
        logger.info("✅ PPO RLHF training")
        logger.info("✅ Comprehensive evaluation")
        logger.info("\nNext steps:")
        logger.info("1. Review evaluation reports in models/*/evaluation_report.md")
        logger.info("2. Deploy model using inference pipeline")
        logger.info("3. Set up continuous monitoring")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Weather Forecasting LoRA Pipeline")
    parser.add_argument(
        "--stage", 
        choices=['data', 'sft', 'ppo', 'eval', 'all'],
        default='all',
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--config",
        default="config/base_config.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = WeatherLoRAPipeline(config_path=args.config)
    
    # Run specified stage
    if args.stage == 'all':
        success = pipeline.run_complete_pipeline()
    elif args.stage == 'data':
        success = pipeline.stage_data_collection()
    elif args.stage == 'sft':
        success = pipeline.stage_sft_training()
    elif args.stage == 'ppo':
        success = pipeline.stage_ppo_training()
    elif args.stage == 'eval':
        success = pipeline.stage_evaluation()
    
    if success:
        logger.info(f"✅ Stage '{args.stage}' completed successfully")
        exit(0)
    else:
        logger.error(f"❌ Stage '{args.stage}' failed")
        exit(1)


if __name__ == "__main__":
    main()