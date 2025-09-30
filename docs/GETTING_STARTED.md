# Getting Started with Weather Forecasting LoRA

This guide will help you get started with the Weather Forecasting LoRA project, implementing the complete methodology from Schulman et al. (2025).

## Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd "weather forecasting"

# Activate virtual environment
.\weather-lora-env\Scripts\Activate.ps1

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

The easiest way to get started is to run the complete pipeline:

```bash
# Run all stages (data collection → SFT → PPO → evaluation)
python run_complete_pipeline.py --stage all

# Or run individual stages:
python run_complete_pipeline.py --stage data
python run_complete_pipeline.py --stage sft  
python run_complete_pipeline.py --stage ppo
python run_complete_pipeline.py --stage eval
```

### 3. Quick Data Collection Test

```bash
# Test data collection
python collect_sample_data.py
```

## Step-by-Step Guide

### Phase 1: Data Collection (Week 1)

1. **Collect Weather Data**:
   ```bash
   python collect_sample_data.py
   ```
   
   This will:
   - Fetch current weather data from Open-Meteo API
   - Gather historical data for training
   - Preprocess into training sequences
   - Create train/val/test splits

2. **Verify Data**:
   ```bash
   # Check data files
   ls data/raw/
   ls data/processed/
   
   # View statistics
   cat data/processed/dataset_stats.json
   ```

### Phase 2: SFT Training (Week 2-3)

1. **Configure Training**:
   - Edit `config/sft_config.yaml` if needed
   - Adjust batch sizes based on your GPU memory

2. **Train SFT Model**:
   ```bash
   python train_sft.py --config config/sft_config.yaml
   ```
   
   This implements Schulman et al. methodology:
   - LoRA adapters on all linear layers
   - 10x learning rate scaling
   - Frozen base weights
   - Moderate batch sizes

3. **Monitor Training**:
   ```bash
   # View logs
   tensorboard --logdir logs/
   ```

### Phase 3: PPO Training (Week 4-5)

1. **Run PPO Training**:
   ```bash
   python run_complete_pipeline.py --stage ppo
   ```
   
   This implements:
   - Composite reward model (accuracy + style + calibration)
   - KL regularization to prevent drift
   - Value head + LoRA adapters only

### Phase 4: Evaluation (Week 6)

1. **Comprehensive Evaluation**:
   ```bash
   python run_complete_pipeline.py --stage eval
   ```
   
   Generates:
   - Performance metrics (BLEU, ROUGE, accuracy)
   - Evaluation reports
   - Model comparison

2. **View Results**:
   ```bash
   # Check evaluation reports
   cat models/weather-lora-sft/evaluation_report.md
   cat models/weather-lora-ppo/evaluation_report.md
   
   # View comparison
   cat evaluation_results.json
   ```

## Individual Component Usage

### Data Collection

```python
from src.data import WeatherDataCollector, MAJOR_CITIES

# Initialize collector
collector = WeatherDataCollector()

# Fetch data for major cities
observations = collector.fetch_open_meteo_current(
    locations=MAJOR_CITIES[:3],
    days_ahead=7
)

# Save data
collector.save_data(observations, "weather_data.csv")
```

### LoRA Training

```python
from src.models import WeatherForecasterLoRA, LoRATrainer

# Initialize model
model = WeatherForecasterLoRA(
    base_model_name="microsoft/Mistral-7B-v0.1",
    quantization=True
)

# Train
trainer = LoRATrainer(model)
trainer.train(train_dataset, eval_dataset)
```

### Inference

```python
from src.inference import WeatherInference
from src.data import WeatherLocation

# Initialize inference
inference = WeatherInference("models/weather-lora-ppo")
inference.load_model()

# Generate forecast
location = WeatherLocation("New York", 40.7128, -74.0060)
forecast = inference.process_request(ForecastRequest(location=location))
print(forecast.forecast_text)
```

### Evaluation

```python
from src.evaluation import WeatherEvaluator

# Initialize evaluator
evaluator = WeatherEvaluator()

# Evaluate model
metrics = evaluator.evaluate_model(model, test_dataset)

# Generate report
report = evaluator.create_evaluation_report(metrics)
print(report)
```

## Configuration

### Key Configuration Files

- `config/base_config.yaml` - Base model and LoRA settings
- `config/sft_config.yaml` - Supervised fine-tuning parameters
- `config/ppo_config.yaml` - PPO and RLHF settings

### Important Parameters

**LoRA Configuration** (following Schulman et al.):
```yaml
lora:
  r: 32                    # Rank
  alpha: 32               # Alpha scaling  
  dropout: 0.05           # Light dropout
  target_modules:         # All linear layers
    - "q_proj"
    - "k_proj" 
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**Training Configuration**:
```yaml
training:
  learning_rate: 5e-5     # 10x FullFT LR (Schulman scaling)
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch = 32
  num_train_epochs: 3
```

**PPO Configuration**:
```yaml
ppo:
  learning_rate: 1e-5     # Smaller than SFT
  kl_penalty: "kl"        # KL regularization  
  init_kl_coef: 0.1       # KL coefficient
  batch_size: 8           # Moderate batch size
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config files
   - Enable gradient checkpointing
   - Use smaller model variant

2. **Data Collection Fails**:
   - Check internet connection
   - Verify API rate limits
   - Try fewer locations

3. **Training Slow**:
   - Enable quantization (4-bit)
   - Use gradient accumulation
   - Reduce max_length

4. **Model Performance Poor**:
   - Increase training data
   - Adjust learning rates
   - Tune reward weights in PPO

### Performance Tuning

**For 8GB GPU**:
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  fp16: true
  gradient_checkpointing: true
```

**For 16GB GPU**:
```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  fp16: true
```

**For 24GB+ GPU**:
```yaml
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  fp16: true
```

## Next Steps

1. **Customize for Your Domain**:
   - Add more weather variables
   - Include regional patterns
   - Extend to other forecast types

2. **Scale Up Training**:
   - Use more training data
   - Experiment with larger models
   - Add domain-specific metrics

3. **Deploy to Production**:
   - Set up FastAPI server
   - Add monitoring and logging
   - Implement A/B testing

4. **Advanced Features**:
   - Multi-modal inputs (satellite images)
   - Ensemble forecasting
   - Uncertainty quantification

## Resources

- [Project Documentation](README.md)
- [Schulman et al. (2025) Paper](https://thinkingmachines.ai/blog/lora/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Weather API Documentation](https://open-meteo.com/en/docs)

For questions and support, please check the logs and error messages, or refer to the comprehensive documentation in the repository.
