"""
Training Script for Weather Forecasting LoRA
============================================

This script implements the complete training pipeline for Phase 1: SFT (Supervised Fine-Tuning)
following Schulman et al. (2025) methodology.

Usage:
    python train_sft.py --config config/sft_config.yaml
"""

import sys
import argparse
import logging
import json
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lora_model import WeatherForecasterLoRA, LoRATrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(dataset_path: str) -> list:
    """Load training dataset from JSON file."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        if dataset_path.endswith('.jsonl'):
            dataset = []
            for line in f:
                dataset.append(json.loads(line.strip()))
        else:
            dataset = json.load(f)
    
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Weather Forecasting LoRA model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sft_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override base model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        help="Override training data path"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        help="Override validation data path"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config['model']['base_model'] = args.model
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.train_data:
        config['data']['train_path'] = args.train_data
    if args.val_data:
        config['data']['val_path'] = args.val_data
    
    # Print configuration summary
    logger.info("=== Training Configuration ===")
    logger.info(f"Base Model: {config['model']['base_model']}")
    logger.info(f"LoRA Rank: {config['lora']['r']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Batch Size: {config['training']['per_device_train_batch_size']} x {config['training']['gradient_accumulation_steps']}")
    logger.info(f"Epochs: {config['training']['num_train_epochs']}")
    logger.info(f"Output Dir: {config['training']['output_dir']}")
    logger.info("=" * 50)
    
    # Load datasets
    train_dataset = load_dataset(config['data']['train_path'])
    val_dataset = None
    if Path(config['data']['val_path']).exists():
        val_dataset = load_dataset(config['data']['val_path'])
    else:
        logger.warning("Validation dataset not found, training without validation")
    
    # Initialize model
    logger.info("Initializing WeatherForecasterLoRA model...")
    model = WeatherForecasterLoRA(
        base_model_name=config['model']['base_model'],
        lora_config=config['lora'],
        quantization=config['model'].get('quantization', True),
        device_map=config['model'].get('device_map', 'auto')
    )
    
    # Initialize trainer
    trainer = LoRATrainer(model=model, config_path=args.config)
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available, training will be slow")
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config_save_path = output_dir / "training_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Training configuration saved to {config_save_path}")
    
    try:
        # Start training
        logger.info("🚀 Starting LoRA fine-tuning training...")
        logger.info("Following Schulman et al. (2025) methodology:")
        logger.info("  ✓ Frozen base weights")
        logger.info("  ✓ LoRA adapters on all linear layers")
        logger.info("  ✓ 10x learning rate scaling")
        logger.info("  ✓ Moderate batch size for stability")
        
        training_history = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            output_dir=str(output_dir)
        )
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Training history saved to: {history_path}")
        
        # Print final metrics
        if training_history:
            final_metrics = training_history[-1]
            logger.info("Final training metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
        
        # Generate sample forecast
        logger.info("\n=== Sample Forecast Generation ===")
        sample_prompt = """Weather data for New York on 2025-09-30 12:00 UTC:
- Temperature (°C): 23.0, 24.0, 22.0, 21.0
- Humidity (%): 70, 75, 80, 82
- Wind speed (km/h): 12.0, 18.0, 20.0, 15.0
- Pressure (hPa): 1008.0, 1007.0, 1006.0, 1005.0
- Precipitation probability: 0.10, 0.20, 0.60, 0.70

Generate a forecast bulletin:"""
        
        try:
            sample_forecast = model.generate_forecast(sample_prompt)
            logger.info("Sample generated forecast:")
            logger.info(sample_forecast)
        except Exception as e:
            logger.warning(f"Could not generate sample forecast: {e}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("=== Next Steps ===")
    logger.info("1. Evaluate model performance on test set")
    logger.info("2. Proceed to Phase 2: RLHF with PPO")
    logger.info("3. Deploy model for inference")


if __name__ == "__main__":
    main()