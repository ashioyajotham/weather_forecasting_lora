# Weather Forecasting with LoRA Fine-tuning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive implementation of weather forecasting using LoRA (Low-Rank Adaptation) fine-tuning on Large Language Models, following Schulman et al. (2025) methodology.

## 🌤️ Project Overview

This project transforms numerical weather data into natural language forecasts using state-of-the-art LoRA fine-tuning techniques. It implements a complete pipeline from data collection to deployment, following the "LoRA Without Regret" methodology.

### Key Features

- **Numerical → Text Mapping**: Convert structured weather data to natural language forecasts
- **LoRA Fine-tuning**: Efficient adaptation with frozen base weights  
- **RLHF with PPO**: Optimize forecasts for accuracy and style
- **Modular Architecture**: Composable adapters for different forecasting domains
- **Comprehensive Evaluation**: Accuracy, calibration, and style metrics

## 📁 Project Structure

```
weather-forecasting/
├── src/                    # Core source code
│   ├── data/              # Data collection & preprocessing
│   ├── models/            # LoRA models & training
│   ├── evaluation/        # Metrics & evaluation
│   ├── rl/               # Reinforcement learning components
│   ├── inference/        # Deployment & API
│   └── utils/            # Configuration & utilities
├── data/                  # Raw & processed datasets
├── models/               # Trained model checkpoints
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks for analysis
├── tests/                # Unit tests
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv weather-lora-env
.\weather-lora-env\Scripts\activate  # Windows
# source weather-lora-env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection

```python
from src.data import WeatherDataCollector

# Initialize data collector
collector = WeatherDataCollector()

# Fetch ERA5 reanalysis data
era5_data = collector.fetch_era5(
    start_date="2020-01-01",
    end_date="2023-12-31",
    variables=["temperature", "humidity", "pressure", "wind_speed"]
)

# Fetch Open-Meteo forecasts
forecasts = collector.fetch_open_meteo(
    locations=["New York", "London", "Tokyo"],
    days_back=365
)
```

### 3. Training LoRA Model

```python
from src.models import WeatherForecasterLoRA, LoRATrainer

# Initialize model with LoRA configuration
model = WeatherForecasterLoRA(
    base_model="microsoft/Mistral-7B-v0.1",
    lora_config={
        "r": 32,
        "alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "dropout": 0.05
    }
)

# Train with supervised fine-tuning
trainer = LoRATrainer(model=model, config="config/sft_config.yaml")
trainer.train(train_dataset, eval_dataset)
```

### 4. RLHF with PPO

```python
from src.rl import PPOTrainerWeather, RewardModel

# Load SFT model and add value head
ppo_model = model.add_value_head()

# Define reward model
reward_model = RewardModel(accuracy_weight=0.7, style_weight=0.3)

# PPO training
ppo_trainer = PPOTrainerWeather(
    model=ppo_model,
    reward_model=reward_model,
    config="config/ppo_config.yaml"
)
ppo_trainer.train()
```

### 5. Inference

```python
from src.inference import WeatherInference

# Load trained model
inference = WeatherInference("models/weather-lora-ppo")

# Generate forecast
weather_input = {
    "location": "New York",
    "temperature": [23, 24, 22, 21],
    "humidity": [70, 75, 80, 82],
    "wind_speed": [12, 18, 20, 15],
    "precipitation_probability": [0.1, 0.2, 0.6, 0.7]
}

forecast = inference.generate_forecast(weather_input)
print(forecast)
# Output: "Afternoon temperatures around 23-24°C with high humidity. 
#          Winds increasing to 20 kph by early evening. 
#          Showers likely by evening with 60%+ precipitation chances."
```

## 📊 Training Schedule

Following the 8-week schedule from the project specification:

- **Week 1**: Data Setup & Baseline
- **Week 2-3**: Phase 1 SFT with LoRA  
- **Week 4-5**: Phase 2 RL with PPO
- **Week 6**: Robustness & Ablations
- **Week 7**: Deployment Prep
- **Week 8+**: Continuous Feedback Loop

## 🎯 Methodology Alignment

This implementation strictly follows Schulman et al. (2025) "LoRA Without Regret":

✅ **Frozen base weights** with LoRA adapters only  
✅ **All linear layers** (attention + MLP)  
✅ **10× learning rate scaling** for LoRA  
✅ **KL regularization** in PPO phase  
✅ **Moderate batch sizes** for stability  
✅ **Modular adapters** for deployment  

## 📈 Evaluation Metrics

- **Accuracy**: Categorical prediction (rain/no-rain, temperature bands)
- **Calibration**: Brier score for probability predictions  
- **Style**: BLEU/ROUGE vs human forecasts
- **Readability**: Human evaluation scores
- **Factual Consistency**: Comparison with observed weather

## 🛠️ Configuration

All configurations are stored in `config/` directory:

- `base_config.yaml`: Base model and general settings
- `sft_config.yaml`: Supervised fine-tuning parameters
- `ppo_config.yaml`: PPO and RLHF settings
- `data_config.yaml`: Data sources and preprocessing
- `eval_config.yaml`: Evaluation metrics and thresholds

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_data.py
pytest tests/test_models.py
pytest tests/test_evaluation.py
```

## 📚 Documentation

- [API Reference](docs/api.md)
- [Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](docs/contributing.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Schulman et al. (2025) - "LoRA Without Regret" methodology
- Thinking Machines Lab - Inspiration and guidance
- Hugging Face - Transformers, PEFT, and TRL libraries
- European Centre for Medium-Range Weather Forecasts (ECMWF) - ERA5 data
- Open-Meteo - Weather API services

## 📞 Contact

For questions and support, please open an issue or contact the development team.

---

*Built with ❤️ for advancing AI-powered weather forecasting*