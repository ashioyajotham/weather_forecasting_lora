# Contributing to Weather Forecasting LoRA

Thank you for your interest in contributing to the Weather Forecasting LoRA project! This is a research implementation following Schulman et al. (2025) "LoRA Without Regret" methodology, and we welcome contributions that maintain scientific rigor and advance the state of weather forecasting with LLMs.

## 🎯 Project Vision

This project aims to advance the intersection of Large Language Models and meteorological forecasting through parameter-efficient fine-tuning. We're building a research-grade implementation that serves both as a practical tool and a foundation for future research.

## 🤝 Types of Contributions

We welcome several types of contributions:

### 🔬 Research Contributions

- **Methodology improvements** following Schulman et al. guidelines
- **Novel evaluation metrics** for weather forecasting quality
- **Ablation studies** and experimental analysis
- **Performance optimizations** for LoRA training
- **Multi-modal extensions** (satellite imagery, radar data)

### 💻 Technical Contributions

- **Bug fixes** and stability improvements
- **Documentation** enhancements and examples
- **Test coverage** expansion
- **API improvements** and new endpoints
- **Deployment optimizations**

### 📊 Data Contributions

- **New weather data sources** and integrations
- **Data preprocessing** improvements
- **Evaluation datasets** for benchmarking
- **Regional adaptations** and localization

### 📚 Documentation Contributions

- **Tutorial notebooks** and examples
- **API documentation** improvements
- **Research methodology** explanations
- **Deployment guides** for different platforms

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (32GB recommended)

### Environment Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ashioyajotham/weather_forecasting_lora.git
   cd weather_forecasting_lora
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv weather-lora-env
   # Windows
   .\weather-lora-env\Scripts\activate
   # Linux/Mac
   source weather-lora-env/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # Development dependencies
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**:

   ```bash
   pre-commit install
   ```

5. **Verify installation**:

   ```bash
   python -m pytest tests/ -v
   python collect_sample_data.py --verify
   ```

## 📋 Development Workflow

### Branch Strategy

- **`main`**: Stable, research-ready code
- **`develop`**: Integration branch for new features
- **`feature/*`**: Individual feature development
- **`experiment/*`**: Research experiments and ablations
- **`hotfix/*`**: Critical bug fixes

### Workflow Steps

1. **Create a new branch**:

   ```bash
   git checkout -b feature/your-feature-name
   # or for experiments
   git checkout -b experiment/ablation-study-name
   ```

2. **Make your changes**:
   - Follow the coding standards (see below)
   - Add appropriate tests
   - Update documentation

3. **Test your changes**:

   ```bash
   # Run all tests
   python -m pytest tests/ -v
   
   # Run specific test suites
   python -m pytest tests/test_models.py
   python -m pytest tests/test_data.py
   
   # Run lint checks
   flake8 src/
   black --check src/
   ```

4. **Commit your changes**:

   ```bash
   git add .
   git commit -m "feat: add new weather data source integration"
   ```

5. **Push and create PR**:

   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import ordering**: isort configuration in `pyproject.toml`
- **Docstrings**: Google style docstrings
- **Type hints**: Required for all public functions

### Code Formatting

We use automated formatting tools:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Example Code Style

```python
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer

class WeatherForecaster:
    """Weather forecasting model using LoRA fine-tuning.
    
    This class implements the Schulman et al. (2025) methodology for
    parameter-efficient fine-tuning of large language models.
    
    Args:
        model_name: Name of the base model to fine-tune
        lora_config: LoRA configuration parameters
        device: Device to run the model on
        
    Examples:
        >>> forecaster = WeatherForecaster(
        ...     model_name="microsoft/Mistral-7B-v0.1",
        ...     lora_config={"r": 32, "alpha": 32}
        ... )
        >>> forecast = forecaster.predict(weather_data)
    """
    
    def __init__(
        self,
        model_name: str,
        lora_config: Dict[str, int],
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.lora_config = lora_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    def predict(
        self, 
        weather_data: Dict[str, List[float]]
    ) -> Tuple[str, float]:
        """Generate weather forecast from numerical data.
        
        Args:
            weather_data: Dictionary containing weather variables
            
        Returns:
            Tuple of (forecast_text, confidence_score)
            
        Raises:
            ValueError: If weather_data is malformed
        """
        # Implementation here
        pass
```

### Documentation Standards

- **Module docstrings**: Brief description and usage examples
- **Class docstrings**: Purpose, args, examples
- **Function docstrings**: Args, returns, raises, examples
- **Inline comments**: For complex logic and research decisions

## 🧪 Testing Guidelines

### Test Structure

```text
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for workflows
├── performance/       # Performance and benchmark tests
├── fixtures/          # Test data and fixtures
└── conftest.py       # Pytest configuration
```

### Test Types

1. **Unit Tests**: Test individual functions and classes

   ```python
   def test_weather_data_preprocessing():
       """Test weather data preprocessing pipeline."""
       raw_data = {"temperature": [20, 22, 21], "humidity": [65, 70, 68]}
       processed = preprocess_weather_data(raw_data)
       assert processed["prompt"].startswith("Weather conditions:")
       assert len(processed["tokens"]) > 0
   ```

2. **Integration Tests**: Test complete workflows

   ```python
   def test_end_to_end_training():
       """Test complete training pipeline."""
       config = load_test_config()
       trainer = LoRATrainer(config)
       metrics = trainer.train(test_dataset)
       assert metrics["loss"] < 2.0
       assert metrics["bleu_score"] > 0.1
   ```

3. **Performance Tests**: Benchmark training and inference

   ```python
   def test_inference_speed():
       """Test inference latency requirements."""
       model = load_trained_model()
       start_time = time.time()
       forecast = model.predict(sample_weather_data)
       inference_time = time.time() - start_time
       assert inference_time < 1.0  # Must be under 1 second
   ```

### Test Data

- Use **synthetic data** for unit tests
- Use **small real datasets** for integration tests
- **Mock external APIs** in tests
- Include **edge cases** and error conditions

## 🔬 Research Contributions Guidelines

### Experimental Design

When contributing research improvements:

1. **Hypothesis**: Clearly state what you're testing
2. **Methodology**: Follow or extend Schulman et al. (2025) principles
3. **Baselines**: Compare against existing implementation
4. **Metrics**: Use established meteorological and NLP metrics
5. **Reproducibility**: Provide complete experimental setup

### Experiment Documentation

Create detailed documentation for experiments:

```markdown
## Experiment: Multi-City Training Ablation

### Hypothesis
Training on weather data from multiple cities improves forecast 
generalization compared to single-city training.

### Methodology
- Base model: Mistral-7B
- LoRA config: r=32, α=32, following Schulman et al.
- Training data: 50 cities vs 1 city (New York)
- Evaluation: BLEU score and meteorological accuracy

### Results
| Setup | BLEU | Accuracy | Training Time |
|-------|------|----------|---------------|
| Single city | 0.42 | 0.78 | 2h |
| Multi-city | 0.48 | 0.83 | 6h |

### Conclusion
Multi-city training improves generalization with acceptable 
training overhead.
```

### Research Artifacts

Include all experimental artifacts:

- **Configuration files** for reproducibility
- **Training logs** and metrics
- **Model checkpoints** (if significant)
- **Evaluation results** and analysis
- **Jupyter notebooks** with analysis

## 📚 Documentation Standards

### README Updates

When adding new features, update relevant documentation:

- Feature description in main README
- Usage examples
- Configuration options
- Performance implications

### API Documentation

Use docstrings that generate clear API docs:

```python
def train_lora_model(
    config: TrainingConfig,
    dataset: WeatherDataset,
    callbacks: Optional[List[Callback]] = None,
) -> TrainingResults:
    """Train LoRA model following Schulman et al. methodology.
    
    This function implements the complete training pipeline including
    data preprocessing, LoRA adapter initialization, and training loop
    with proper learning rate scaling.
    
    Args:
        config: Training configuration with LoRA parameters
        dataset: Weather forecasting dataset
        callbacks: Optional training callbacks for logging/monitoring
        
    Returns:
        TrainingResults object containing metrics and model paths
        
    Example:
        >>> config = TrainingConfig(learning_rate=5e-5, lora_r=32)
        >>> dataset = WeatherDataset.from_csv("weather_data.csv")
        >>> results = train_lora_model(config, dataset)
        >>> print(f"Final loss: {results.final_loss}")
    """
```

## 🚀 Pull Request Guidelines

### PR Title Format

Use conventional commits format:

- `feat: add new weather API integration`
- `fix: resolve memory leak in training loop`
- `docs: update API documentation`
- `test: add integration tests for PPO training`
- `perf: optimize LoRA adapter initialization`
- `refactor: restructure data preprocessing pipeline`

### PR Description Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Research experiment
- [ ] Breaking change

## Methodology Compliance
- [ ] Follows Schulman et al. (2025) guidelines
- [ ] Maintains LoRA parameter efficiency
- [ ] Preserves model modularity

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass (if applicable)
- [ ] Manual testing completed

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] README updated (if needed)
- [ ] Examples provided

## Research Impact
If this is a research contribution:
- Methodology changes: [describe]
- Performance impact: [quantify]
- Reproducibility: [provide details]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated checks**: All CI/CD checks must pass
2. **Code review**: At least one maintainer review required
3. **Research review**: Research contributions need methodology review
4. **Documentation review**: Ensure clarity and completeness

## 🐛 Bug Reports

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug and expected behavior.

## Reproduction Steps
1. Step one
2. Step two
3. Error occurs

## Environment
- OS: [e.g., Windows 11, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 2.0.1]
- CUDA version: [e.g., 11.8]
- Package versions: [from pip freeze]

## Error Logs
```text
Paste error logs here
```

## Additional Context

Any other relevant information.

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature.

## Research Motivation
How does this advance weather forecasting with LLMs?

## Implementation Approach
High-level approach for implementation.

## Methodology Alignment
How does this align with Schulman et al. principles?

## Success Criteria
How will we measure success?

## Alternatives Considered
Other approaches considered and why this is preferred.
```

## 🔒 Security Guidelines

### Data Privacy

- **No personal data** in code or datasets
- **Anonymize location data** when possible
- **Secure API keys** - never commit credentials
- **Use environment variables** for sensitive configuration

### Code Security

- **Validate inputs** for all external data
- **Sanitize outputs** before logging
- **Use secure dependencies** - regularly update packages
- **Follow security best practices** for ML models

## 🏆 Recognition

### Contributors

All contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Paper acknowledgments** (for research contributions)
- **Release notes** for significant contributions

### Research Contributions

Significant research contributions may be eligible for:

- **Co-authorship** on research papers
- **Conference presentation** opportunities
- **Research collaboration** invitations

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and research discussions
- **Email**: [maintainer email] for sensitive issues

### Documentation Resources

- **Getting Started Guide**: `GETTING_STARTED.md`
- **API Documentation**: `docs/api.md`
- **Research Methodology**: `Training Recipe for LoRA in Weather Forecasting.md`
- **Project Status**: `PROJECT_STATUS.md`

### Development Support

- **Code examples**: Check `notebooks/` directory
- **Test examples**: Review `tests/` directory
- **Configuration examples**: See `config/` directory

## 📄 License and Attribution

This project is licensed under the MIT License with research attribution requirements. By contributing, you agree that your contributions will be licensed under the same terms.

### Research Attribution

When contributing research improvements, ensure proper attribution to:

- **Schulman et al. (2025)** - Core LoRA methodology
- **Original authors** of any algorithms or techniques used
- **Data sources** - Weather data providers and APIs

---

Thank you for contributing to weather forecasting research! Your contributions help advance the intersection of Large Language Models and meteorological science. 🌤️
