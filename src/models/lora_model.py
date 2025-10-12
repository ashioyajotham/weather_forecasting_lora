"""
LoRA Model Implementation for Weather Forecasting
=================================================

This module implements the LoRA fine-tuning approach following Schulman et al. (2025).
Key features:
- Frozen base weights with LoRA adapters
- All linear layers adaptation (attention + MLP)
- Proper learning rate scaling (10x FullFT LR)
- Weather-specific prompt formatting

Following the "LoRA Without Regret" methodology.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset
import logging
from typing import Dict, List, Optional, Union
import yaml
from pathlib import Path

# W&B integration
from src.utils.wandb_logger import WandBLogger, WandBCallback

logger = logging.getLogger(__name__)


class WeatherForecasterLoRA:
    """
    Weather forecasting model using LoRA fine-tuning.

    Implements the methodology from Schulman et al. (2025):
    - LoRA adapters on all linear layers
    - Frozen base model weights
    - Efficient adaptation for weather domain
    """

    def __init__(
        self,
        base_model_name: str = "mistralai/Mistral-7B-v0.1",
        lora_config: Optional[Dict] = None,
        quantization: bool = True,
        device_map: str = "auto",
    ):
        """
        Initialize WeatherForecasterLoRA model.

        Args:
            base_model_name: Hugging Face model name
            lora_config: LoRA configuration dictionary
            quantization: Whether to use 4-bit quantization
            device_map: Device mapping strategy
        """
        self.base_model_name = base_model_name
        self.device_map = device_map
        self.quantization = quantization

        # Default LoRA config following Schulman et al.
        self.lora_config = lora_config or {
            "r": 32,  # Rank (Schulman recommends 16-64)
            "alpha": 32,  # Alpha scaling = 32
            "dropout": 0.05,  # Light dropout
            "bias": "none",  # No bias adaptation
            "task_type": TaskType.CAUSAL_LM,
            # Target all linear layers (Schulman Sec. 2.2)
            "target_modules": [
                "q_proj",  # Query projection
                "k_proj",  # Key projection
                "v_proj",  # Value projection
                "o_proj",  # Output projection
                "gate_proj",  # Gate projection (MLP)
                "up_proj",  # Up projection (MLP)
                "down_proj",  # Down projection (MLP)
            ],
        }

        self.model = None
        self.tokenizer = None
        self.peft_model = None

        logger.info(f"Initialized WeatherForecasterLoRA with {base_model_name}")

    def load_model(self):
        """Load base model and apply LoRA configuration."""
        logger.info("Loading base model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True, use_fast=True
        )

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Configure quantization if enabled
        quantization_config = None
        if self.quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Prepare model for training if quantized
        if self.quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA configuration
        lora_config = LoraConfig(**self.lora_config)
        self.peft_model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.peft_model.print_trainable_parameters()

        logger.info("Model loaded successfully with LoRA adapters")
        return self.peft_model

    def prepare_dataset(self, dataset: List[Dict], max_length: int = 512) -> Dataset:
        """
        Prepare dataset for training.

        Args:
            dataset: List of training examples
            max_length: Maximum sequence length

        Returns:
            Hugging Face Dataset object
        """
        logger.info(f"Preparing dataset with {len(dataset)} examples")

        def tokenize_function(examples):
            """Tokenize input-target pairs."""
            # Combine prompt and response for causal LM training
            texts = []
            for input_text, target_text in zip(examples["input"], examples["target"]):
                # Format as instruction-following conversation
                formatted_text = (
                    f"{input_text}\n{target_text}{self.tokenizer.eos_token}"
                )
                texts.append(formatted_text)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Convert to Hugging Face dataset
        hf_dataset = Dataset.from_list(dataset)

        # Tokenize dataset
        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=hf_dataset.column_names,
            desc="Tokenizing dataset",
        )

        logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset

    def create_training_arguments(
        self,
        output_dir: str = "./models/weather-lora-sft",
        config_path: Optional[str] = None,
    ) -> TrainingArguments:
        """
        Create training arguments following Schulman et al. methodology.

        Args:
            output_dir: Directory to save model outputs
            config_path: Path to training configuration file

        Returns:
            TrainingArguments object
        """
        # Default training configuration (Schulman recommendations)
        default_config = {
            "output_dir": output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 8,  # Effective batch size = 32
            "learning_rate": 5e-5,  # 10x higher than typical FullFT (Schulman Sec. 3.1)
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 1.0,
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_pin_memory": True,
            "eval_strategy": "steps",
            "eval_steps": 500,
            "save_strategy": "steps",
            "save_steps": 1000,
            "logging_steps": 100,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": ["tensorboard"],
            "run_name": "weather-lora-sft",
            "seed": 42,
            "remove_unused_columns": False,
            "ddp_find_unused_parameters": False,
        }

        # Load config from file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if "training" in config:
                    default_config.update(config["training"])

        # Create TrainingArguments
        training_args = TrainingArguments(**default_config)

        logger.info(
            f"Training arguments created for {training_args.num_train_epochs} epochs"
        )
        logger.info(f"Learning rate: {training_args.learning_rate} (LoRA scaling)")
        logger.info(
            f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
        )

        return training_args

    def get_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[TrainingArguments] = None,
    ) -> Trainer:
        """
        Create Hugging Face Trainer for LoRA fine-tuning.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments

        Returns:
            Trainer object
        """
        if self.peft_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if training_args is None:
            training_args = self.create_training_arguments()

        # Data collator for causal LM
        from transformers import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        logger.info("Trainer created successfully")
        return trainer

    def save_model(self, output_path: str):
        """
        Save the LoRA adapter weights.

        Args:
            output_path: Path to save the adapter
        """
        if self.peft_model is None:
            raise ValueError("No model to save. Train model first.")

        self.peft_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"LoRA adapter saved to {output_path}")

    def load_adapter(self, adapter_path: str):
        """
        Load a trained LoRA adapter.

        Args:
            adapter_path: Path to the saved adapter
        """
        if self.model is None:
            self.load_model()

        from peft import PeftModel

        self.peft_model = PeftModel.from_pretrained(
            self.model, adapter_path, is_trainable=False
        )

        logger.info(f"LoRA adapter loaded from {adapter_path}")
        return self.peft_model

    def generate_forecast(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate weather forecast from prompt.

        Args:
            prompt: Input weather data prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated forecast text
        """
        if self.peft_model is None:
            raise ValueError(
                "Model not loaded. Call load_model() or load_adapter() first."
            )

        # Tokenize input
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.peft_model.device)

        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and extract generated part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt) :].strip()

        return generated_text


class LoRATrainer:
    """
    High-level trainer class for weather forecasting LoRA models.

    Handles the complete training pipeline with proper configurations
    following Schulman et al. methodology, including W&B experiment tracking.
    """

    def __init__(
        self,
        model: WeatherForecasterLoRA,
        config_path: Optional[str] = None,
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize LoRA trainer.

        Args:
            model: WeatherForecasterLoRA instance
            config_path: Path to configuration file
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name (overrides config)
            wandb_run_name: W&B run name (overrides config)
        """
        self.model = model
        self.config_path = config_path
        self.trainer = None
        self.use_wandb = use_wandb
        
        # Initialize W&B logger if enabled
        self.wandb_logger = None
        if self.use_wandb:
            self.wandb_logger = WandBLogger(
                config_path=config_path,
                project=wandb_project,
                name=wandb_run_name,
            )
            logger.info("W&B logging enabled")
        
        logger.info("LoRATrainer initialized")

    def train(
        self,
        train_dataset: List[Dict],
        eval_dataset: Optional[List[Dict]] = None,
        output_dir: str = "./models/weather-lora-sft",
    ):
        """
        Complete training pipeline with W&B tracking.

        Args:
            train_dataset: Training data
            eval_dataset: Evaluation data
            output_dir: Output directory
        """
        logger.info("Starting LoRA training pipeline...")

        # Load model if not already loaded
        if self.model.peft_model is None:
            self.model.load_model()

        # Initialize W&B run if enabled
        if self.use_wandb and self.wandb_logger:
            # Load full config for W&B
            full_config = {}
            if self.config_path and Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
            
            # Add training metadata
            full_config.update({
                'base_model': self.model.base_model_name,
                'lora_r': self.model.lora_config['r'],
                'lora_alpha': self.model.lora_config['alpha'],
                'train_samples': len(train_dataset),
                'eval_samples': len(eval_dataset) if eval_dataset else 0,
                'quantization': self.model.quantization,
                'methodology': 'Schulman et al. (2025)',
            })
            
            # Initialize W&B
            self.wandb_logger.init(config=full_config)
            
            # Watch model for gradient/parameter tracking
            if self.model.peft_model:
                self.wandb_logger.watch_model(
                    self.model.peft_model,
                    log_freq=self.wandb_logger.config.get('watch_freq', 1000),
                    log='all' if self.wandb_logger.config.get('log_gradients', True) else 'parameters'
                )

        # Prepare datasets
        train_hf_dataset = self.model.prepare_dataset(train_dataset)
        eval_hf_dataset = None
        if eval_dataset:
            eval_hf_dataset = self.model.prepare_dataset(eval_dataset)

        # Create training arguments
        training_args = self.model.create_training_arguments(
            output_dir=output_dir, config_path=self.config_path
        )

        # Create trainer with W&B callback if enabled
        self.trainer = self.model.get_trainer(
            train_dataset=train_hf_dataset,
            eval_dataset=eval_hf_dataset,
            training_args=training_args,
        )
        
        # Add W&B callback
        if self.use_wandb and self.wandb_logger:
            wandb_callback = WandBCallback(self.wandb_logger)
            self.trainer.add_callback(wandb_callback)
            logger.info("W&B callback added to trainer")

        # Start training
        logger.info("🚀 Starting LoRA fine-tuning...")
        train_result = self.trainer.train()

        # Save model
        self.model.save_model(output_dir)
        
        # Log final model artifact to W&B
        if self.use_wandb and self.wandb_logger:
            if self.wandb_logger.config.get('log_artifacts', True):
                self.wandb_logger.log_model_artifact(
                    model_path=output_dir,
                    artifact_name='weather-lora-final',
                    aliases=['final', 'best'],
                    metadata={
                        'final_loss': train_result.training_loss,
                        'epochs': training_args.num_train_epochs,
                        'base_model': self.model.base_model_name,
                    }
                )

        logger.info("✅ Training completed successfully!")
        logger.info(f"Model saved to: {output_dir}")
        
        # Finish W&B run
        if self.use_wandb and self.wandb_logger:
            self.wandb_logger.finish()

        return self.trainer.state.log_history


def main():
    """Example usage of WeatherForecasterLoRA."""
    # Initialize model
    model = WeatherForecasterLoRA(
        base_model_name="microsoft/Mistral-7B-v0.1", quantization=True
    )

    # Load model
    model.load_model()

    # Example training data
    example_data = [
        {
            "input": """Weather data for New York on 2025-09-30 12:00 UTC:
- Temperature (°C): 23.0, 24.0, 22.0, 21.0
- Humidity (%): 70, 75, 80, 82
- Wind speed (km/h): 12.0, 18.0, 20.0, 15.0
- Pressure (hPa): 1008.0, 1007.0, 1006.0, 1005.0
- Precipitation probability: 0.10, 0.20, 0.60, 0.70

Generate a forecast bulletin:""",
            "target": "Afternoon temperatures around 23-24°C with increasing humidity. Winds picking up to 20 km/h by early evening. Showers likely by evening with precipitation chances above 60%.",
        }
    ]

    # Initialize trainer
    trainer = LoRATrainer(model)

    # Train (this would normally use a full dataset)
    # trainer.train(example_data)

    print("Example setup completed. Ready for training with full dataset.")


if __name__ == "__main__":
    main()
