#!/usr/bin/env python3
"""
Weather Forecasting LoRA Training Script
=========================================

Complete training pipeline with W&B experiment tracking.
This script implements the complete training pipeline for Phase 1: SFT (Supervised Fine-Tuning)

Usage:
    python train_lora.py --config config/base_config.yaml --output models/weather-lora-v1

Following Schulman et al. (2025) methodology for LoRA fine-tuning.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Core imports
from src.models.lora_model import WeatherForecasterLoRA, LoRATrainer
from src.evaluation.metrics import WeatherEvaluator
from src.utils.wandb_logger import WandBLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_dataset(data_path: str) -> List[Dict]:
    """
    Load training/validation/test dataset.
    
    Args:
        data_path: Path to JSON dataset file
        
    Returns:
        List of examples with 'input' and 'target' fields
    """
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    return data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Weather Forecasting LoRA model with W&B tracking'
    )
    
    # Data arguments
    parser.add_argument(
        '--train_data',
        type=str,
        default='data/processed/train.json',
        help='Path to training data JSON file'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default='data/processed/val.json',
        help='Path to validation data JSON file'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='data/processed/test.json',
        help='Path to test data JSON file'
    )
    
    # Model arguments
    parser.add_argument(
        '--base_model',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='Base model name from Hugging Face'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to configuration YAML file'
    )
    
    # Training arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/weather-lora-sft',
        help='Output directory for model checkpoints'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Training batch size (overrides config)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    # W&B arguments
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='W&B project name (overrides config)'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='W&B run name (auto-generated if not specified)'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable W&B logging'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only run evaluation on test set (no training)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to trained model for evaluation'
    )
    
    # Hardware arguments
    parser.add_argument(
        '--quantization',
        action='store_true',
        default=True,
        help='Use 4-bit quantization (default: True)'
    )
    parser.add_argument(
        '--no_quantization',
        action='store_true',
        help='Disable quantization'
    )
    
    return parser.parse_args()


def main():
    """Main training/evaluation pipeline."""
    args = parse_args()
    
    # Print configuration
    logger.info("="*80)
    logger.info("Weather Forecasting LoRA Training")
    logger.info("Following Schulman et al. (2025) Methodology")
    logger.info("="*80)
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"W&B Enabled: {not args.no_wandb}")
    logger.info("="*80)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    if not args.eval_only:
        train_data = load_dataset(args.train_data)
        val_data = load_dataset(args.val_data) if Path(args.val_data).exists() else None
    
    test_data = load_dataset(args.test_data) if Path(args.test_data).exists() else None
    
    # Initialize model
    logger.info("Initializing WeatherForecasterLoRA...")
    model = WeatherForecasterLoRA(
        base_model_name=args.base_model,
        quantization=not args.no_quantization if args.no_quantization else args.quantization
    )
    
    # Evaluation-only mode
    if args.eval_only:
        if not args.model_path:
            logger.error("--model_path required for --eval_only mode")
            sys.exit(1)
        
        logger.info(f"Loading model from {args.model_path}")
        model.load_model()
        model.load_adapter(args.model_path)
        
        if test_data:
            logger.info("Running evaluation on test set...")
            evaluator = WeatherEvaluator()
            metrics = evaluator.evaluate_model(model, test_data)
            
            # Print results
            logger.info("\n" + "="*80)
            logger.info("EVALUATION RESULTS")
            logger.info("="*80)
            logger.info(f"BLEU Score: {metrics.bleu_score:.4f}")
            logger.info(f"ROUGE-1 F1: {metrics.rouge_1_f:.4f}")
            logger.info(f"ROUGE-2 F1: {metrics.rouge_2_f:.4f}")
            logger.info(f"ROUGE-L F1: {metrics.rouge_l_f:.4f}")
            logger.info(f"Temperature MAE: {metrics.temperature_mae:.2f}°C")
            logger.info(f"Wind Speed MAE: {metrics.wind_speed_mae:.2f} km/h")
            logger.info(f"Rain Accuracy: {metrics.rain_accuracy:.4f}")
            logger.info(f"Overall Score: {metrics.overall_score:.4f}")
            logger.info("="*80)
            
            # Save report
            report_path = Path(args.output_dir) / 'evaluation_report.md'
            evaluator.create_evaluation_report(metrics, str(report_path))
            logger.info(f"Report saved to {report_path}")
        
        return
    
    # Training mode
    logger.info("Initializing LoRATrainer...")
    trainer = LoRATrainer(
        model=model,
        config_path=args.config,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
    # Train model
    logger.info("Starting training...")
    try:
        training_history = trainer.train(
            train_dataset=train_data,
            eval_dataset=val_data,
            output_dir=args.output_dir
        )
        
        logger.info("✅ Training completed successfully!")
        
        # Save training history
        history_path = Path(args.output_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Post-training evaluation
    if test_data:
        logger.info("\n" + "="*80)
        logger.info("Running post-training evaluation...")
        logger.info("="*80)
        
        # Initialize evaluator with W&B logger if available
        wandb_logger = trainer.wandb_logger if not args.no_wandb else None
        evaluator = WeatherEvaluator(wandb_logger=wandb_logger)
        
        # Run evaluation
        metrics = evaluator.evaluate_model(
            model,
            test_data,
            step=None,  # Final evaluation
            log_to_wandb=not args.no_wandb
        )
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("FINAL EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"BLEU Score: {metrics.bleu_score:.4f}")
        logger.info(f"ROUGE-1 F1: {metrics.rouge_1_f:.4f}")
        logger.info(f"ROUGE-2 F1: {metrics.rouge_2_f:.4f}")
        logger.info(f"ROUGE-L F1: {metrics.rouge_l_f:.4f}")
        logger.info(f"Temperature MAE: {metrics.temperature_mae:.2f}°C")
        logger.info(f"Wind Speed MAE: {metrics.wind_speed_mae:.2f} km/h")
        logger.info(f"Rain Accuracy: {metrics.rain_accuracy:.4f}")
        logger.info(f"Overall Score: {metrics.overall_score:.4f}")
        logger.info("="*80)
        
        # Save evaluation report
        report_path = Path(args.output_dir) / 'final_evaluation_report.md'
        evaluator.create_evaluation_report(metrics, str(report_path))
        logger.info(f"Evaluation report saved to {report_path}")
    
    logger.info("\n🎉 All done! Model ready for deployment.")


if __name__ == '__main__':
    main()
