"""
Merge LoRA Adapter and Convert to GGUF
======================================

1. Merge LoRA adapter with TinyLlama base model
2. Save merged model
3. Convert to GGUF format for llama.cpp inference
"""

import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "models/weather-lora-peft/lora_adapter"
MERGED_PATH = "models/weather-merged"


def merge_lora():
    """Merge LoRA adapter with base model."""
    logger.info("=" * 80)
    logger.info("Merging LoRA Adapter with Base Model")
    logger.info("=" * 80)
    
    # Load base model
    logger.info(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # Merge LoRA with base model
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    output_path = Path(MERGED_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Merge Complete!")
    logger.info("=" * 80)
    logger.info(f"Merged model saved to: {output_path}")
    logger.info("")
    logger.info("Next: Convert to GGUF using llama.cpp")
    logger.info("Run: python llama.cpp/convert_hf_to_gguf.py models/weather-merged --outfile models/gguf/weather-tinyllama.gguf --outtype f16")
    
    return output_path


if __name__ == "__main__":
    merge_lora()
