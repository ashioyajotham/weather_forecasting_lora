"""
Weather LoRA Training with Transformers/PEFT
=============================================

CPU-based LoRA training using Hugging Face transformers and PEFT.
After training, convert to GGUF for llama.cpp inference.
"""

import os
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "output_dir": "models/weather-lora-peft",
    "train_data": "data/processed/train.json",
    
    # LoRA config
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    
    # Training config (optimized for CPU)
    "num_epochs": 1,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_length": 512,
    "warmup_ratio": 0.03,
    "save_steps": 100,
    "logging_steps": 10,
}


def load_training_data(path: str, max_samples: int = None):
    """Load and format training data."""
    logger.info(f"Loading training data from {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    # Format for Mistral instruction format
    formatted = []
    for item in data:
        text = f"<s>[INST] {item['input']} [/INST] {item['target']} </s>"
        formatted.append({"text": text})
    
    logger.info(f"Loaded {len(formatted)} training examples")
    return Dataset.from_list(formatted)


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    logger.info("=" * 80)
    logger.info("Weather LoRA Training with Transformers/PEFT")
    logger.info("=" * 80)
    logger.info("")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info("")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model (TinyLlama is small enough for CPU training ~2GB)
    logger.info(f"Loading model: {CONFIG['model_name']}")
    logger.info("TinyLlama-1.1B fits in ~2GB RAM...")
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_name'],
        torch_dtype=torch.float32,  # CPU uses float32
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")


    
    # Configure LoRA
    logger.info("")
    logger.info("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and tokenize data
    logger.info("")
    train_dataset = load_training_data(
        CONFIG['train_data'],
        max_samples=1000,  # Limit for CPU training
    )
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, CONFIG['max_length']),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        learning_rate=CONFIG['learning_rate'],
        warmup_ratio=CONFIG['warmup_ratio'],
        logging_steps=CONFIG['logging_steps'],
        save_steps=CONFIG['save_steps'],
        save_total_limit=2,
        fp16=False,  # CPU doesn't support fp16
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # For CPU
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("")
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    logger.info(f"Epochs: {CONFIG['num_epochs']}")
    logger.info(f"Batch size: {CONFIG['batch_size']}")
    logger.info(f"Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    logger.info(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    logger.info("")
    
    start_time = datetime.now()
    trainer.train()
    
    # Save
    logger.info("")
    logger.info("Saving LoRA adapter...")
    model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")
    
    elapsed = datetime.now() - start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Time: {elapsed}")
    logger.info(f"LoRA adapter saved to: {output_dir / 'lora_adapter'}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Merge LoRA with base model")
    logger.info("2. Convert to GGUF for llama.cpp")


if __name__ == "__main__":
    main()
