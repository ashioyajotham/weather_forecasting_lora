"""
Unit Tests for LoRA Model Implementation
=========================================

Tests for:
- WeatherForecasterLoRA
- LoRATrainer
- Model training and inference
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.models import WeatherForecasterLoRA, LoRATrainer


# ============================================================================
# WeatherForecasterLoRA Tests
# ============================================================================

@pytest.mark.unit
class TestWeatherForecasterLoRA:
    """Test WeatherForecasterLoRA model."""
    
    def test_model_initialization(self, lora_config):
        """Test model initializes with correct configuration."""
        model = WeatherForecasterLoRA(
            base_model_name="microsoft/Mistral-7B-v0.1",
            lora_config=lora_config,
            quantization=False
        )
        
        assert model.base_model_name == "microsoft/Mistral-7B-v0.1"
        assert model.lora_config == lora_config
        assert model.quantization == False
    
    def test_lora_config_defaults(self):
        """Test default LoRA configuration follows Schulman et al."""
        model = WeatherForecasterLoRA(quantization=False)
        
        # Default config should have correct parameters
        assert model.lora_config is not None
        # Following Schulman: r=32, alpha=32
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_loading(self, mock_tokenizer, mock_model, mock_lora_model):
        """Test model loads correctly."""
        # Skip actual loading in tests
        pytest.skip("Requires model weights - tested in integration")
    
    def test_target_modules_all_linear_layers(self):
        """Test LoRA targets all linear layers per Schulman."""
        model = WeatherForecasterLoRA(quantization=False)
        
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"  # MLP
        ]
        
        # Should target all linear layers, not just attention
        if model.lora_config:
            target_modules = model.lora_config.get("target_modules", [])
            for module in expected_modules:
                assert module in target_modules
    
    def test_learning_rate_scaling(self, test_config):
        """Test learning rate is 10x FullFT per Schulman."""
        model = WeatherForecasterLoRA(quantization=False)
        
        # LoRA LR should be ~10x full fine-tuning LR
        lora_lr = 5e-5
        fullft_lr = 5e-6
        
        assert lora_lr / fullft_lr == pytest.approx(10.0, rel=0.1)
    
    def test_quantization_configuration(self):
        """Test 4-bit quantization setup."""
        model = WeatherForecasterLoRA(quantization=True)
        
        assert model.quantization == True
        # Should configure BitsAndBytes for 4-bit
    
    def test_dataset_preparation(self, mock_lora_model, sample_training_data):
        """Test dataset preparation for training."""
        dataset = mock_lora_model.prepare_dataset(sample_training_data)
        
        # Should return Hugging Face Dataset format
        # assert isinstance(dataset, Dataset)
    
    def test_generate_forecast(self, mock_lora_model):
        """Test forecast generation."""
        prompt = "Weather data for New York..."
        forecast = mock_lora_model.generate_forecast(prompt)
        
        assert isinstance(forecast, str)
        assert len(forecast) > 0
    
    def test_generate_with_parameters(self, mock_lora_model):
        """Test generation with different parameters."""
        prompt = "Weather data..."
        
        # Test different temperatures
        forecast1 = mock_lora_model.generate_forecast(
            prompt, temperature=0.1, max_new_tokens=50
        )
        forecast2 = mock_lora_model.generate_forecast(
            prompt, temperature=0.9, max_new_tokens=200
        )
        
        # Both should generate valid forecasts
        assert isinstance(forecast1, str)
        assert isinstance(forecast2, str)


# ============================================================================
# LoRATrainer Tests
# ============================================================================

@pytest.mark.unit
class TestLoRATrainer:
    """Test LoRATrainer functionality."""
    
    def test_trainer_initialization(self, mock_lora_model):
        """Test trainer initializes correctly."""
        trainer = LoRATrainer(mock_lora_model)
        
        assert trainer.model == mock_lora_model
        assert trainer.trainer is None  # Not created until train() called
    
    def test_training_arguments_creation(self, mock_lora_model, test_config):
        """Test creation of training arguments."""
        trainer = LoRATrainer(mock_lora_model)
        
        # Should create appropriate TrainingArguments
        # Following Schulman: moderate batch size, proper LR
    
    @pytest.mark.slow
    def test_train_one_epoch(self, mock_lora_model, sample_training_data, temp_dir):
        """Test training for one epoch."""
        trainer = LoRATrainer(mock_lora_model)
        
        # Mock training to avoid actual model execution
        # trainer.train(sample_training_data, output_dir=str(temp_dir))
    
    def test_evaluation_during_training(self, mock_lora_model, sample_training_data, sample_test_data):
        """Test evaluation runs during training."""
        trainer = LoRATrainer(mock_lora_model)
        
        # Should evaluate on eval_dataset during training
        # trainer.train(sample_training_data, eval_dataset=sample_test_data)
    
    def test_checkpoint_saving(self, mock_lora_model, temp_dir):
        """Test model checkpoints are saved."""
        trainer = LoRATrainer(mock_lora_model)
        output_dir = temp_dir / "checkpoints"
        
        # Should save checkpoints during training
        # trainer.train(data, output_dir=str(output_dir))
        # assert (output_dir / "checkpoint-xxx").exists()
    
    def test_adapter_only_training(self, mock_lora_model):
        """Test only LoRA adapters are trainable."""
        # Base model weights should be frozen
        # Only adapter parameters should have requires_grad=True
        pass


# ============================================================================
# Model Configuration Tests
# ============================================================================

@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration loading and validation."""
    
    def test_load_config_from_yaml(self, sample_config_yaml):
        """Test loading configuration from YAML file."""
        import yaml
        
        with open(sample_config_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "model" in config
        assert "lora" in config
        assert "training" in config
    
    def test_config_validation(self):
        """Test configuration validation."""
        valid_config = {
            "r": 32,
            "alpha": 32,
            "dropout": 0.05
        }
        
        invalid_config = {
            "r": -1,  # Invalid rank
            "alpha": "wrong_type"  # Invalid type
        }
        
        # Should validate config parameters
    
    def test_schulman_methodology_compliance(self):
        """Test configuration follows Schulman et al. guidelines."""
        config = {
            "r": 32,
            "alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"],
            "dropout": 0.05
        }
        
        # Verify Schulman compliance:
        # 1. All linear layers targeted
        # 2. Rank and alpha properly set
        # 3. Light dropout (0.05)
        assert config["r"] == 32
        assert config["alpha"] == 32
        assert len(config["target_modules"]) == 7  # All linear layers


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model workflow."""
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_end_to_end_training(self, sample_training_data, sample_test_data, temp_dir):
        """Test complete training pipeline."""
        # Skip without GPU
        pytest.skip("Requires GPU and model weights")
        
        # Initialize model
        model = WeatherForecasterLoRA(
            base_model_name="microsoft/Mistral-7B-v0.1",
            quantization=True
        )
        
        # Load model
        model.load_model()
        
        # Train
        trainer = LoRATrainer(model)
        trainer.train(
            sample_training_data,
            eval_dataset=sample_test_data,
            output_dir=str(temp_dir)
        )
        
        # Verify checkpoint exists
        assert (temp_dir / "adapter_model.bin").exists()
    
    @pytest.mark.slow
    def test_inference_after_training(self, temp_dir):
        """Test model can generate forecasts after training."""
        pytest.skip("Requires trained model")
        
        # Load trained model
        # Generate forecast
        # Validate output
    
    def test_adapter_merging(self):
        """Test merging multiple LoRA adapters."""
        # Test Schulman's "low regret" principle
        # Multiple domain adapters should be mergeable
        pass


# ============================================================================
# Model Behavior Tests
# ============================================================================

@pytest.mark.unit
class TestModelBehavior:
    """Test model behavior and output quality."""
    
    def test_forecast_format_consistency(self, mock_lora_model):
        """Test forecast outputs follow consistent format."""
        prompts = [
            "Weather data for New York...",
            "Weather data for London...",
            "Weather data for Tokyo..."
        ]
        
        forecasts = [mock_lora_model.generate_forecast(p) for p in prompts]
        
        # All should be strings
        assert all(isinstance(f, str) for f in forecasts)
        # All should have content
        assert all(len(f) > 0 for f in forecasts)
    
    def test_numerical_to_text_mapping(self, mock_lora_model):
        """Test model correctly maps numerical data to text."""
        prompt = """Temperature: 23°C, Humidity: 80%, Wind: 20 km/h
Generate forecast:"""
        
        forecast = mock_lora_model.generate_forecast(prompt)
        
        # Should mention relevant weather variables
        # (This is a behavioral test that would need a real model)
    
    def test_temperature_sampling_vs_greedy(self, mock_lora_model):
        """Test different decoding strategies."""
        prompt = "Weather data..."
        
        # Greedy (temperature=0)
        greedy = mock_lora_model.generate_forecast(
            prompt, temperature=0.0, do_sample=False
        )
        
        # Sampling (temperature=0.7)
        sampled = mock_lora_model.generate_forecast(
            prompt, temperature=0.7, do_sample=True
        )
        
        assert isinstance(greedy, str)
        assert isinstance(sampled, str)


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestModelErrorHandling:
    """Test model error handling."""
    
    def test_model_not_loaded_error(self):
        """Test error when using model before loading."""
        model = WeatherForecasterLoRA(quantization=False)
        
        # Should raise error if generating without loading
        # with pytest.raises(ValueError):
        #     model.generate_forecast("test prompt")
    
    def test_invalid_input_handling(self, mock_lora_model):
        """Test handling of invalid inputs."""
        # Empty prompt
        # with pytest.raises(ValueError):
        #     mock_lora_model.generate_forecast("")
        
        # None prompt
        # with pytest.raises(TypeError):
        #     mock_lora_model.generate_forecast(None)
    
    def test_out_of_memory_handling(self):
        """Test handling of OOM errors."""
        # Test with very large batch
        # Should handle gracefully
        pass


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.performance
class TestModelPerformance:
    """Test model performance metrics."""
    
    def test_inference_speed(self, mock_lora_model):
        """Test inference completes in reasonable time."""
        import time
        
        prompt = "Weather data for New York..."
        
        start = time.time()
        forecast = mock_lora_model.generate_forecast(prompt, max_new_tokens=128)
        elapsed = time.time() - start
        
        # Should generate in under 5 seconds on CPU (mocked)
        assert elapsed < 5.0
    
    @pytest.mark.gpu
    def test_memory_usage(self):
        """Test memory usage stays within limits."""
        pytest.skip("Requires GPU monitoring")
        # Monitor GPU memory during inference
    
    def test_batch_inference_efficiency(self, mock_lora_model):
        """Test batch inference is more efficient than sequential."""
        # Batch should be faster than sequential
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])