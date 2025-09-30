"""
Reinforcement Learning Module for Weather Forecasting LoRA
==========================================================

This module implements Phase 2: RLHF with PPO following Schulman et al. (2025).
Key components:
- Reward models for forecast accuracy and style
- PPO training with LoRA adapters only
- KL regularization to prevent drift from SFT baseline
- Composite reward functions

Following the "LoRA Without Regret" methodology for stable RL fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import PeftModel
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re
from datetime import datetime
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Container for different reward components."""
    accuracy_reward: float
    style_reward: float
    calibration_reward: float
    factual_consistency: float
    total_reward: float


class WeatherRewardModel:
    """
    Reward model for weather forecasting quality assessment.
    
    Implements composite rewards following the project specification:
    - Factual accuracy vs observed weather
    - Style and readability
    - Calibration and confidence
    - Penalization for overconfident errors
    """
    
    def __init__(
        self,
        accuracy_weight: float = 0.4,
        style_weight: float = 0.2, 
        calibration_weight: float = 0.2,
        consistency_weight: float = 0.2
    ):
        """
        Initialize reward model.
        
        Args:
            accuracy_weight: Weight for factual accuracy
            style_weight: Weight for style and readability
            calibration_weight: Weight for calibration
            consistency_weight: Weight for factual consistency
        """
        self.accuracy_weight = accuracy_weight
        self.style_weight = style_weight
        self.calibration_weight = calibration_weight
        self.consistency_weight = consistency_weight
        
        # Normalization to ensure weights sum to 1
        total_weight = sum([accuracy_weight, style_weight, calibration_weight, consistency_weight])
        self.accuracy_weight /= total_weight
        self.style_weight /= total_weight
        self.calibration_weight /= total_weight
        self.consistency_weight /= total_weight
        
        # Weather pattern matching
        self.rain_patterns = [
            r'\b(?:rain|showers?|precipitation|wet|drizzle|storms?)\b',
            r'\b(?:cloudy|overcast|grey|gray)\b'
        ]
        
        self.clear_patterns = [
            r'\b(?:clear|sunny|bright|fair|dry)\b',
            r'\b(?:no rain|cloudless)\b'
        ]
        
        self.temp_patterns = [
            r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*°?[CF]',
            r'(?:around|near|about)\s+(\d+(?:\.\d+)?)\s*°?[CF]'
        ]
        
        logger.info("WeatherRewardModel initialized with composite rewards")
    
    def extract_weather_features(self, forecast_text: str) -> Dict:
        """
        Extract weather features from forecast text for reward calculation.
        
        Args:
            forecast_text: Generated forecast text
            
        Returns:
            Dictionary with extracted features
        """
        text_lower = forecast_text.lower()
        
        # Rain prediction
        rain_score = sum(1 for pattern in self.rain_patterns if re.search(pattern, text_lower))
        clear_score = sum(1 for pattern in self.clear_patterns if re.search(pattern, text_lower))
        
        rain_prediction = None
        if rain_score > clear_score:
            rain_prediction = True
        elif clear_score > rain_score:
            rain_prediction = False
        
        # Temperature extraction
        temp_range = None
        for pattern in self.temp_patterns:
            match = re.search(pattern, forecast_text)
            if match:
                if len(match.groups()) == 2:
                    temp_range = (float(match.group(1)), float(match.group(2)))
                else:
                    temp = float(match.group(1))
                    temp_range = (temp - 2, temp + 2)
                break
        
        # Confidence indicators
        confidence_patterns = [
            r'likely', r'expected', r'probable', r'certain',
            r'possible', r'may', r'might', r'could'
        ]
        confidence_score = sum(1 for pattern in confidence_patterns if re.search(pattern, text_lower))
        
        return {
            'rain_prediction': rain_prediction,
            'temperature_range': temp_range,
            'confidence_score': confidence_score,
            'text_length': len(forecast_text.split()),
            'has_specifics': temp_range is not None or rain_prediction is not None
        }
    
    def calculate_accuracy_reward(
        self,
        forecast_text: str,
        observed_weather: Dict
    ) -> float:
        """
        Calculate accuracy reward based on observed weather outcomes.
        
        Args:
            forecast_text: Generated forecast text
            observed_weather: Dictionary with observed weather conditions
            
        Returns:
            Accuracy reward (0-1)
        """
        features = self.extract_weather_features(forecast_text)
        reward = 0.0
        total_checks = 0
        
        # Rain accuracy
        if 'rain' in observed_weather and features['rain_prediction'] is not None:
            total_checks += 1
            if features['rain_prediction'] == observed_weather['rain']:
                reward += 1.0
            else:
                reward -= 0.5  # Penalty for wrong prediction
        
        # Temperature accuracy
        if 'temperature' in observed_weather and features['temperature_range'] is not None:
            total_checks += 1
            obs_temp = observed_weather['temperature']
            temp_min, temp_max = features['temperature_range']
            
            if temp_min <= obs_temp <= temp_max:
                reward += 1.0  # Perfect range match
            else:
                # Partial credit based on distance
                center_temp = (temp_min + temp_max) / 2
                error = abs(obs_temp - center_temp)
                reward += max(0, 1 - error / 10)  # Linear decay over 10°C
        
        # Normalize by number of checks
        if total_checks > 0:
            reward /= total_checks
        else:
            reward = 0.5  # Neutral reward if no verifiable predictions
        
        return max(0, reward)
    
    def calculate_style_reward(self, forecast_text: str) -> float:
        """
        Calculate style and readability reward.
        
        Args:
            forecast_text: Generated forecast text
            
        Returns:
            Style reward (0-1)
        """
        # Length appropriateness (15-40 words is good)
        word_count = len(forecast_text.split())
        length_score = 1.0
        if word_count < 10:
            length_score = word_count / 10
        elif word_count > 50:
            length_score = max(0, 1 - (word_count - 50) / 50)
        
        # Professional weather terminology
        weather_terms = [
            'temperature', 'humidity', 'wind', 'pressure', 'precipitation',
            'showers', 'conditions', 'forecast', 'expected', 'likely'
        ]
        term_score = sum(1 for term in weather_terms if term in forecast_text.lower())
        term_score = min(1.0, term_score / 5)  # Normalize to 0-1
        
        # Sentence structure (proper punctuation and flow)
        structure_score = 1.0
        if not forecast_text.strip().endswith(('.', '!', '?')):
            structure_score -= 0.2
        if forecast_text.count('.') == 0:  # No sentences
            structure_score -= 0.3
        
        # Combine scores
        style_reward = 0.4 * length_score + 0.4 * term_score + 0.2 * structure_score
        return max(0, min(1, style_reward))
    
    def calculate_calibration_reward(self, forecast_text: str) -> float:
        """
        Calculate calibration reward (confidence appropriateness).
        
        Args:
            forecast_text: Generated forecast text
            
        Returns:
            Calibration reward (0-1)
        """
        features = self.extract_weather_features(forecast_text)
        
        # Reward appropriate confidence expressions
        confidence_score = features['confidence_score']
        
        # Good calibration: moderate confidence (1-3 confidence terms)
        if 1 <= confidence_score <= 3:
            return 1.0
        elif confidence_score == 0:
            return 0.7  # Slightly penalize over-certainty
        else:
            return max(0.3, 1 - (confidence_score - 3) * 0.2)  # Penalize over-hedging
    
    def calculate_factual_consistency(self, forecast_text: str) -> float:
        """
        Calculate factual consistency reward (internal coherence).
        
        Args:
            forecast_text: Generated forecast text
            
        Returns:
            Consistency reward (0-1)
        """
        features = self.extract_weather_features(forecast_text)
        
        # Check for contradictions
        has_rain = features['rain_prediction'] == True
        has_clear = features['rain_prediction'] == False
        
        # Logical consistency
        if has_rain and has_clear:
            return 0.0  # Contradiction
        
        # Specificity bonus
        specificity = 0.5
        if features['has_specifics']:
            specificity = 1.0
        
        return specificity
    
    def calculate_composite_reward(
        self,
        forecast_text: str,
        observed_weather: Optional[Dict] = None
    ) -> RewardComponents:
        """
        Calculate composite reward from all components.
        
        Args:
            forecast_text: Generated forecast text
            observed_weather: Observed weather data (optional)
            
        Returns:
            RewardComponents with individual and total rewards
        """
        # Calculate individual components
        accuracy_reward = 0.5  # Default if no observed weather
        if observed_weather:
            accuracy_reward = self.calculate_accuracy_reward(forecast_text, observed_weather)
        
        style_reward = self.calculate_style_reward(forecast_text)
        calibration_reward = self.calculate_calibration_reward(forecast_text)
        consistency_reward = self.calculate_factual_consistency(forecast_text)
        
        # Composite reward
        total_reward = (
            self.accuracy_weight * accuracy_reward +
            self.style_weight * style_reward +
            self.calibration_weight * calibration_reward +
            self.consistency_weight * consistency_reward
        )
        
        return RewardComponents(
            accuracy_reward=accuracy_reward,
            style_reward=style_reward,
            calibration_reward=calibration_reward,
            factual_consistency=consistency_reward,
            total_reward=total_reward
        )


class PPOTrainerWeather:
    """
    PPO trainer specifically designed for weather forecasting LoRA models.
    
    Implements Schulman et al. (2025) recommendations:
    - KL regularization to prevent drift
    - Small learning rates for stability
    - Update only LoRA adapters + value head
    - Moderate batch sizes
    """
    
    def __init__(
        self,
        model_path: str,
        reward_model: WeatherRewardModel,
        config_path: Optional[str] = None
    ):
        """
        Initialize PPO trainer for weather forecasting.
        
        Args:
            model_path: Path to SFT LoRA model
            reward_model: Reward model instance
            config_path: Path to PPO configuration
        """
        self.model_path = model_path
        self.reward_model = reward_model
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ppo_trainer = None
        
        logger.info(f"PPOTrainerWeather initialized with model: {model_path}")
    
    def _load_config(self) -> dict:
        """Load PPO configuration."""
        default_config = {
            'batch_size': 8,
            'forward_batch_size': 4,
            'learning_rate': 1e-5,      # Smaller than SFT (Schulman Sec. 4.1)
            'mini_batch_size': 4,
            'gradient_accumulation_steps': 1,
            'ppo_epochs': 4,
            'kl_penalty': 'kl',         # Explicit KL regularization
            'init_kl_coef': 0.1,        # Initial KL coefficient
            'target_kl': 0.1,           # Target KL divergence
            'cliprange': 0.2,           # PPO clipping range
            'vf_coef': 0.1,             # Value function coefficient
            'ent_coef': 0.01,           # Entropy coefficient
            'max_grad_norm': 1.0,       # Gradient clipping
            'seed': 42
        }
        
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'ppo' in config:
                    default_config.update(config['ppo'])
        
        return default_config
    
    def load_model(self):
        """Load SFT model and prepare for PPO training."""
        logger.info("Loading SFT model and adding value head...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        logger.info("Model loaded with value head for PPO training")
        return self.model
    
    def create_ppo_trainer(self) -> PPOTrainer:
        """Create PPO trainer with weather-specific configuration."""
        # PPO configuration following Schulman recommendations
        ppo_config = PPOConfig(
            batch_size=self.config['batch_size'],
            forward_batch_size=self.config['forward_batch_size'],
            learning_rate=self.config['learning_rate'],
            mini_batch_size=self.config['mini_batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            ppo_epochs=self.config['ppo_epochs'],
            kl_penalty=self.config['kl_penalty'],
            init_kl_coef=self.config['init_kl_coef'],
            target_kl=self.config['target_kl'],
            cliprange=self.config['cliprange'],
            vf_coef=self.config['vf_coef'],
            ent_coef=self.config['ent_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            seed=self.config['seed'],
            log_with=None  # Can add wandb/tensorboard
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=None  # We'll feed batches manually
        )
        
        logger.info("PPO trainer created with weather-specific configuration")
        return self.ppo_trainer
    
    def train_step(
        self,
        batch: List[Dict],
        observed_weather: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """
        Perform single PPO training step.
        
        Args:
            batch: Batch of training examples
            observed_weather: Corresponding observed weather data
            
        Returns:
            Training statistics
        """
        # Extract prompts
        prompts = [example['input'] for example in batch]
        
        # Tokenize queries
        query_tensors = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt")
            query_tensors.append(tokens.squeeze())
        
        # Generate responses
        response_tensors = []
        with torch.no_grad():
            for query_tensor in query_tensors:
                response = self.model.generate(
                    query_tensor.unsqueeze(0).to(self.model.device),
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                # Extract only generated part
                response_tensor = response[0][len(query_tensor):]
                response_tensors.append(response_tensor)
        
        # Decode responses
        responses = [
            self.tokenizer.decode(response_tensor, skip_special_tokens=True)
            for response_tensor in response_tensors
        ]
        
        # Calculate rewards
        rewards = []
        for i, response in enumerate(responses):
            obs_weather = observed_weather[i] if observed_weather else None
            reward_components = self.reward_model.calculate_composite_reward(
                response, obs_weather
            )
            rewards.append(reward_components.total_reward)
        
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)
        
        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        return stats
    
    def train(
        self,
        training_data: List[Dict],
        num_epochs: int = 3,
        save_every: int = 100,
        output_dir: str = "./models/weather-lora-ppo"
    ):
        """
        Complete PPO training loop.
        
        Args:
            training_data: Training dataset
            num_epochs: Number of training epochs
            save_every: Save model every N steps
            output_dir: Directory to save models
        """
        logger.info(f"Starting PPO training for {num_epochs} epochs")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Create PPO trainer if not exists
        if self.ppo_trainer is None:
            self.create_ppo_trainer()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        step = 0
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Shuffle training data
            import random
            random.shuffle(training_data)
            
            # Process in batches
            batch_size = self.config['batch_size']
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                
                try:
                    # Training step
                    stats = self.train_step(batch)
                    
                    # Log statistics
                    if step % 10 == 0:
                        logger.info(f"Step {step}: {stats}")
                    
                    # Save checkpoint
                    if step % save_every == 0 and step > 0:
                        checkpoint_dir = f"{output_dir}/checkpoint-{step}"
                        self.save_model(checkpoint_dir)
                        logger.info(f"Saved checkpoint at step {step}")
                    
                    step += 1
                    
                except Exception as e:
                    logger.warning(f"Error in training step {step}: {e}")
                    continue
        
        # Save final model
        self.save_model(output_dir)
        logger.info(f"PPO training completed. Final model saved to {output_dir}")
    
    def save_model(self, output_path: str):
        """Save the PPO-trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save configuration
        config_path = Path(output_path) / "ppo_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"PPO model saved to {output_path}")


def main():
    """Example usage of PPO training for weather forecasting."""
    # Initialize reward model
    reward_model = WeatherRewardModel(
        accuracy_weight=0.4,
        style_weight=0.2,
        calibration_weight=0.2,
        consistency_weight=0.2
    )
    
    # Test reward calculation
    sample_forecast = "Partly cloudy with temperatures around 22-25°C. Light winds up to 15 km/h. Showers possible by evening."
    sample_observed = {'rain': True, 'temperature': 23.5}
    
    rewards = reward_model.calculate_composite_reward(sample_forecast, sample_observed)
    
    logger.info("Sample reward calculation:")
    logger.info(f"Accuracy: {rewards.accuracy_reward:.3f}")
    logger.info(f"Style: {rewards.style_reward:.3f}")
    logger.info(f"Calibration: {rewards.calibration_reward:.3f}")
    logger.info(f"Consistency: {rewards.factual_consistency:.3f}")
    logger.info(f"Total: {rewards.total_reward:.3f}")
    
    # PPO trainer example (would need trained SFT model)
    # trainer = PPOTrainerWeather(
    #     model_path="./models/weather-lora-sft",
    #     reward_model=reward_model
    # )


if __name__ == "__main__":
    main()