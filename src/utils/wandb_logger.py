"""
Weights & Biases Logging Utilities for Weather Forecasting LoRA

This module provides a comprehensive wrapper for W&B logging functionality,
including metrics tracking, artifact management, and prediction visualization.

Following Schulman et al. (2025) recommendations for experiment tracking.
"""

import wandb
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
import pandas as pd
from datetime import datetime


class WandBLogger:
    """
    Unified W&B logging interface for weather forecasting experiments.
    
    Handles:
    - Experiment initialization and configuration
    - Training/evaluation metrics logging
    - Model artifact management
    - Prediction visualization
    - System metrics tracking
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize W&B logger.
        
        Args:
            config_path: Path to YAML config file with wandb section
            config_dict: Dictionary with wandb configuration
            **kwargs: Additional W&B init parameters (override config)
        """
        self.config = self._load_config(config_path, config_dict)
        self.config.update(kwargs)  # Allow runtime overrides
        self.run = None
        self.is_initialized = False
        
    def _load_config(
        self,
        config_path: Optional[str],
        config_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Load W&B configuration from file or dict."""
        if config_path:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('wandb', {})
        elif config_dict:
            return config_dict.get('wandb', config_dict)
        else:
            # Default minimal config
            return {
                'project': 'weather-forecasting-lora',
                'log_model': 'checkpoint',
                'log_freq': 100
            }
    
    def init(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> wandb.run:
        """
        Initialize W&B run.
        
        Args:
            run_name: Custom run name (overrides config)
            config: Training/model configuration to log
            **kwargs: Additional wandb.init parameters
            
        Returns:
            W&B run object
        """
        # Prepare init parameters
        init_params = {
            'project': self.config.get('project', 'weather-forecasting-lora'),
            'entity': self.config.get('entity'),
            'name': run_name or self.config.get('name'),
            'group': self.config.get('group', 'sft-experiments'),
            'tags': self.config.get('tags', []),
            'notes': self.config.get('notes', ''),
            'config': config or {},
            'reinit': self.config.get('reinit', True),
            'resume': self.config.get('resume', 'allow'),
            'save_code': self.config.get('save_code', True),
        }
        
        # Update with runtime overrides
        init_params.update(kwargs)
        
        # Initialize run
        self.run = wandb.init(**init_params)
        self.is_initialized = True
        
        print(f"✅ W&B initialized: {self.run.name} ({self.run.id})")
        print(f"📊 Dashboard: {self.run.url}")
        
        return self.run
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        commit: bool = True
    ):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step (optional, auto-increments if None)
            commit: Whether to commit the metrics immediately
        """
        if not self.is_initialized:
            print("⚠️ W&B not initialized. Call init() first.")
            return
        
        wandb.log(metrics, step=step, commit=commit)
    
    def log_training_metrics(
        self,
        loss: float,
        learning_rate: float,
        epoch: int,
        step: int,
        grad_norm: Optional[float] = None,
        **additional_metrics
    ):
        """
        Log training metrics with consistent naming.
        
        Args:
            loss: Training loss
            learning_rate: Current learning rate
            epoch: Current epoch
            step: Current training step
            grad_norm: Gradient norm (if available)
            **additional_metrics: Any other metrics to log
        """
        metrics = {
            'train/loss': loss,
            'train/learning_rate': learning_rate,
            'train/epoch': epoch,
        }
        
        if grad_norm is not None:
            metrics['train/grad_norm'] = grad_norm
        
        # Add any additional metrics
        for key, value in additional_metrics.items():
            if not key.startswith('train/'):
                key = f'train/{key}'
            metrics[key] = value
        
        self.log_metrics(metrics, step=step)
    
    def log_evaluation_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = 'eval'
    ):
        """
        Log evaluation metrics with namespace prefix.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step
            prefix: Namespace prefix (default: 'eval')
        """
        prefixed_metrics = {
            f'{prefix}/{key}': value 
            for key, value in metrics.items()
        }
        self.log_metrics(prefixed_metrics, step=step)
    
    def log_weather_metrics(
        self,
        temperature_mae: float,
        temperature_accuracy: float,
        wind_speed_mae: float,
        precipitation_accuracy: float,
        step: int,
        **additional_weather_metrics
    ):
        """
        Log weather-specific evaluation metrics.
        
        Args:
            temperature_mae: Mean absolute error for temperature (°C)
            temperature_accuracy: Temperature prediction accuracy (%)
            wind_speed_mae: Mean absolute error for wind speed (km/h)
            precipitation_accuracy: Precipitation prediction accuracy (%)
            step: Training step
            **additional_weather_metrics: Any other weather metrics
        """
        metrics = {
            'weather/temperature_mae': temperature_mae,
            'weather/temperature_accuracy': temperature_accuracy,
            'weather/wind_speed_mae': wind_speed_mae,
            'weather/precipitation_accuracy': precipitation_accuracy,
        }
        
        # Add additional weather metrics
        for key, value in additional_weather_metrics.items():
            if not key.startswith('weather/'):
                key = f'weather/{key}'
            metrics[key] = value
        
        self.log_metrics(metrics, step=step)
    
    def log_nlg_metrics(
        self,
        bleu_score: float,
        rouge_1_f: float,
        rouge_2_f: float,
        rouge_l_f: float,
        step: int,
        **additional_nlg_metrics
    ):
        """
        Log natural language generation metrics.
        
        Args:
            bleu_score: BLEU score
            rouge_1_f: ROUGE-1 F1 score
            rouge_2_f: ROUGE-2 F1 score
            rouge_l_f: ROUGE-L F1 score
            step: Training step
            **additional_nlg_metrics: Any other NLG metrics
        """
        metrics = {
            'nlg/bleu_score': bleu_score,
            'nlg/rouge_1_f': rouge_1_f,
            'nlg/rouge_2_f': rouge_2_f,
            'nlg/rouge_l_f': rouge_l_f,
        }
        
        # Add additional NLG metrics
        for key, value in additional_nlg_metrics.items():
            if not key.startswith('nlg/'):
                key = f'nlg/{key}'
            metrics[key] = value
        
        self.log_metrics(metrics, step=step)
    
    def log_model_artifact(
        self,
        model_path: str,
        artifact_name: str,
        artifact_type: str = 'model',
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log model checkpoint as W&B artifact.
        
        Args:
            model_path: Path to model checkpoint directory
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (default: 'model')
            aliases: List of aliases (e.g., ['latest', 'best'])
            metadata: Additional metadata to attach
        """
        if not self.is_initialized:
            print("⚠️ W&B not initialized. Call init() first.")
            return
        
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata or {}
        )
        
        # Add model files
        artifact.add_dir(model_path)
        
        # Log artifact with aliases
        self.run.log_artifact(artifact, aliases=aliases or [])
        
        print(f"✅ Logged artifact: {artifact_name}")
    
    def log_predictions(
        self,
        predictions: List[str],
        references: List[str],
        inputs: Optional[List[str]] = None,
        step: Optional[int] = None,
        num_samples: int = 10
    ):
        """
        Log sample predictions as W&B table.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            inputs: Input prompts (optional)
            step: Training step
            num_samples: Number of samples to log
        """
        if not self.is_initialized:
            print("⚠️ W&B not initialized. Call init() first.")
            return
        
        # Limit number of samples
        num_samples = min(num_samples, len(predictions))
        
        # Create table data
        columns = ['Prediction', 'Reference']
        if inputs:
            columns.insert(0, 'Input')
        
        data = []
        for i in range(num_samples):
            row = []
            if inputs:
                row.append(inputs[i])
            row.append(predictions[i])
            row.append(references[i])
            data.append(row)
        
        # Log as table
        table = wandb.Table(columns=columns, data=data)
        self.log_metrics({
            'predictions/samples': table
        }, step=step)
    
    def log_prediction_comparison(
        self,
        prompt: str,
        prediction: str,
        reference: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log detailed prediction comparison.
        
        Args:
            prompt: Input prompt
            prediction: Model prediction
            reference: Ground truth
            metrics: Associated metrics for this prediction
            step: Training step
        """
        if not self.is_initialized:
            return
        
        # Create rich comparison
        comparison = {
            'predictions/comparison': wandb.Html(
                f"""
                <div style="font-family: monospace;">
                    <h3>Input Prompt</h3>
                    <pre>{prompt}</pre>
                    <h3>Model Prediction</h3>
                    <pre style="background: #e8f4f8;">{prediction}</pre>
                    <h3>Reference</h3>
                    <pre style="background: #f0f0f0;">{reference}</pre>
                    <h3>Metrics</h3>
                    <ul>
                        {"".join([f"<li><b>{k}</b>: {v:.4f}</li>" for k, v in metrics.items()])}
                    </ul>
                </div>
                """
            )
        }
        
        self.log_metrics(comparison, step=step)
    
    def watch_model(
        self,
        model: torch.nn.Module,
        log_freq: int = 1000,
        log: str = 'all'
    ):
        """
        Watch model for gradient and parameter tracking.
        
        Args:
            model: PyTorch model to watch
            log_freq: How often to log (in steps)
            log: What to log ('gradients', 'parameters', 'all', or None)
        """
        if not self.is_initialized:
            print("⚠️ W&B not initialized. Call init() first.")
            return
        
        if self.config.get('watch_model', True):
            wandb.watch(
                model,
                log=log,
                log_freq=log_freq,
                log_graph=True
            )
            print(f"👁️ Watching model (log={log}, freq={log_freq})")
    
    def log_system_metrics(self):
        """Log system metrics (GPU, CPU, memory)."""
        if self.config.get('log_system_metrics', True):
            # W&B automatically logs system metrics when enabled
            pass
    
    def finish(self):
        """Finish W&B run."""
        if self.is_initialized and self.run:
            self.run.finish()
            print("✅ W&B run finished")
            self.is_initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class WandBCallback:
    """
    Callback for HuggingFace Trainer integration.
    
    Automatically logs training metrics, evaluation results,
    and model checkpoints during training.
    """
    
    def __init__(self, logger: WandBLogger):
        """
        Initialize callback with W&B logger.
        
        Args:
            logger: WandBLogger instance
        """
        self.logger = logger
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging."""
        if logs:
            step = state.global_step
            
            # Separate training and eval metrics
            train_metrics = {}
            eval_metrics = {}
            
            for key, value in logs.items():
                if key.startswith('eval_'):
                    # Remove 'eval_' prefix, will be added back with namespace
                    eval_metrics[key.replace('eval_', '')] = value
                else:
                    train_metrics[key] = value
            
            # Log to W&B
            if train_metrics:
                self.logger.log_metrics(
                    {f'train/{k}': v for k, v in train_metrics.items()},
                    step=step
                )
            
            if eval_metrics:
                self.logger.log_evaluation_metrics(
                    eval_metrics,
                    step=step
                )
    
    def on_save(self, args, state, control, **kwargs):
        """Called when checkpoint is saved."""
        if self.logger.config.get('log_artifacts', True):
            checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
            
            # Check if checkpoint exists
            if Path(checkpoint_path).exists():
                self.logger.log_model_artifact(
                    model_path=checkpoint_path,
                    artifact_name=f"checkpoint-{state.global_step}",
                    aliases=['latest']
                )


def get_wandb_logger(
    config_path: str = "config/base_config.yaml",
    **kwargs
) -> WandBLogger:
    """
    Factory function to create W&B logger.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional parameters for WandBLogger
        
    Returns:
        Configured WandBLogger instance
    """
    return WandBLogger(config_path=config_path, **kwargs)
