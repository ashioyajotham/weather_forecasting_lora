# Weather Forecasting LoRA - Test Suite

Comprehensive test suite for the Weather Forecasting LoRA project, following research-grade testing standards.

## 📋 Test Structure

```
tests/
├── __init__.py              # Test package configuration
├── conftest.py              # Pytest fixtures and configuration
├── test_data.py             # Data collection & preprocessing tests
├── test_models.py           # LoRA model & training tests
├── test_evaluation.py       # Evaluation framework tests
├── test_inference.py        # Inference & deployment tests
└── README.md               # This file
```

## 🚀 Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Files
```bash
# Data tests
pytest tests/test_data.py -v

# Model tests
pytest tests/test_models.py -v

# Evaluation tests
pytest tests/test_evaluation.py -v

# Inference tests
pytest tests/test_inference.py -v
```

### Run by Test Markers
```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"

# Run only GPU tests
pytest tests/ -m gpu

# Run without API-dependent tests
pytest tests/ -m "not api"
```

### Run with Coverage
```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

## 🏷️ Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests for individual components |
| `integration` | Integration tests for workflows |
| `slow` | Slow-running tests (skip with `-m "not slow"`) |
| `gpu` | Tests requiring GPU hardware |
| `api` | Tests requiring external API access |
| `performance` | Performance and benchmark tests |

## 📊 Test Coverage

The test suite covers:

### Data Module (`test_data.py`)
- ✅ Weather data collection from APIs
- ✅ Data preprocessing and formatting
- ✅ Numerical → text conversion
- ✅ Dataset creation and splitting
- ✅ Data validation and error handling
- ✅ Caching mechanisms

### Models Module (`test_models.py`)
- ✅ LoRA model initialization
- ✅ Model configuration (Schulman et al. compliance)
- ✅ Training pipeline execution
- ✅ Forecast generation
- ✅ Adapter management
- ✅ Memory and performance optimization

### Evaluation Module (`test_evaluation.py`)
- ✅ BLEU/ROUGE metric calculation
- ✅ Meteorological accuracy metrics
- ✅ Rain prediction accuracy
- ✅ Temperature/wind MAE
- ✅ Calibration (Brier score)
- ✅ Evaluation report generation

### Inference Module (`test_inference.py`)
- ✅ Real-time inference
- ✅ Batch processing
- ✅ API integration
- ✅ Model versioning
- ✅ Error handling and recovery
- ✅ Performance and latency testing

## 🧪 Test Categories

### Unit Tests (Fast)
Test individual functions and classes in isolation:
```bash
pytest tests/ -m unit --duration=10
```

### Integration Tests (Medium)
Test complete workflows and component interactions:
```bash
pytest tests/ -m integration
```

### Performance Tests (Slow)
Benchmark performance and resource usage:
```bash
pytest tests/ -m performance
```

## 📝 Writing New Tests

### Test Template
```python
import pytest
from src.your_module import YourClass

@pytest.mark.unit
class TestYourClass:
    """Test YourClass functionality."""
    
    def test_initialization(self):
        """Test class initializes correctly."""
        obj = YourClass()
        assert obj is not None
    
    def test_method_behavior(self):
        """Test specific method behavior."""
        obj = YourClass()
        result = obj.your_method(input_data)
        assert result == expected_output
```

### Using Fixtures
```python
def test_with_fixtures(sample_weather_data, mock_lora_model):
    """Test using pytest fixtures."""
    # Fixtures are automatically injected
    assert sample_weather_data is not None
    assert mock_lora_model is not None
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (10, 20),
    (20, 40),
    (30, 60),
])
def test_doubling(input, expected):
    """Test doubling function with multiple inputs."""
    assert double(input) == expected
```

## 🔧 Continuous Integration

Tests are automatically run in CI/CD pipeline:

```yaml
# .github/workflows/tests.yml
- name: Run tests
  run: |
    pytest tests/ -m "not slow and not gpu"
    pytest tests/ --cov=src --cov-report=xml
```

## 📈 Test Metrics

Target metrics for test suite:
- **Coverage**: >80% code coverage
- **Speed**: Unit tests <5 seconds total
- **Reliability**: 100% pass rate on main branch
- **Maintenance**: Tests updated with code changes

## 🐛 Debugging Tests

### Run with verbose output
```bash
pytest tests/ -vv
```

### Stop on first failure
```bash
pytest tests/ -x
```

### Run specific test
```bash
pytest tests/test_data.py::TestWeatherDataCollector::test_collector_initialization
```

### Show print statements
```bash
pytest tests/ -s
```

### Debug with pdb
```bash
pytest tests/ --pdb
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Contributing Guide](../CONTRIBUTING.md#testing-guidelines)
- [Project Documentation](../README.md)

## 🙏 Contributing Tests

When adding new features:
1. **Write tests first** (TDD approach)
2. **Follow existing patterns** in test files
3. **Add appropriate markers** (`@pytest.mark.unit`, etc.)
4. **Update this README** if adding new test categories
5. **Ensure tests pass** before submitting PR

---

For questions about testing, see [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.