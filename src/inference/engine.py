"""
Inference Pipeline for Weather Forecasting LoRA
================================================

This module provides production-ready inference capabilities:
- Real-time weather data fetching
- Forecast generation with LoRA adapters
- Batch processing and API serving
- Model ensembling and confidence estimation

Designed for deployment following the project specification.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
import numpy as np

# Model and preprocessing imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data import WeatherDataCollector, WeatherLocation
from src.evaluation import WeatherEvaluator

logger = logging.getLogger(__name__)


@dataclass
class ForecastRequest:
    """Request for weather forecast generation."""
    location: WeatherLocation
    forecast_horizon: int = 6  # Hours ahead
    include_confidence: bool = True
    model_version: str = "latest"


@dataclass
class ForecastResponse:
    """Response containing generated weather forecast."""
    forecast_text: str
    confidence_score: float
    location: str
    generated_at: datetime
    valid_until: datetime
    model_version: str
    
    # Optional metadata
    input_data: Optional[Dict] = None
    processing_time_ms: Optional[float] = None


class WeatherInference:
    """
    Production inference engine for weather forecasting.
    
    Handles model loading, input preprocessing, forecast generation,
    and confidence estimation.
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "microsoft/Mistral-7B-v0.1",
        device: str = "auto"
    ):
        """
        Initialize weather inference engine.
        
        Args:
            model_path: Path to trained LoRA adapter
            base_model_name: Base model identifier
            device: Device for inference
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = device
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.data_collector = None
        self.evaluator = None
        
        # Model metadata
        self.model_version = self._get_model_version()
        self.load_time = None
        
        logger.info(f"WeatherInference initialized with model: {model_path}")
    
    def _get_model_version(self) -> str:
        """Get model version from metadata."""
        try:
            metadata_path = Path(self.model_path) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('version', 'unknown')
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")
        
        return f"model-{datetime.now().strftime('%Y%m%d')}"
    
    def load_model(self):
        """Load the trained LoRA model for inference."""
        start_time = datetime.now()
        logger.info("Loading model for inference...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                is_trainable=False
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            self.load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def initialize_data_collector(self):
        """Initialize weather data collector."""
        if self.data_collector is None:
            self.data_collector = WeatherDataCollector(cache_session=True)
            logger.info("Data collector initialized")
    
    def fetch_current_weather(self, location: WeatherLocation) -> Dict:
        """
        Fetch current weather data for a location.
        
        Args:
            location: Weather location
            
        Returns:
            Dictionary with current weather data
        """
        if self.data_collector is None:
            self.initialize_data_collector()
        
        try:
            # Fetch current observations
            observations = self.data_collector.fetch_open_meteo_current(
                locations=[location],
                days_ahead=1
            )
            
            if not observations:
                raise ValueError(f"No weather data available for {location.name}")
            
            # Convert to forecast input format
            recent_obs = observations[:4]  # Last 4 hours
            
            weather_data = {
                'location': location.name,
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
                'temperature': [obs.temperature_c for obs in recent_obs],
                'humidity': [obs.humidity_pct for obs in recent_obs],
                'wind_speed': [obs.wind_speed_kph for obs in recent_obs],
                'pressure': [obs.pressure_hpa for obs in recent_obs],
                'precipitation_probability': [obs.precipitation_probability or 0.0 for obs in recent_obs]
            }
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Failed to fetch weather data for {location.name}: {e}")
            raise
    
    def create_prompt(self, weather_data: Dict) -> str:
        """
        Create inference prompt from weather data.
        
        Args:
            weather_data: Dictionary with weather variables
            
        Returns:
            Formatted prompt string
        """
        # Format sequences as comma-separated strings
        temp_str = ", ".join([f"{t:.1f}" for t in weather_data['temperature']])
        humidity_str = ", ".join([f"{h:.0f}" for h in weather_data['humidity']])
        wind_str = ", ".join([f"{w:.1f}" for w in weather_data['wind_speed']])
        pressure_str = ", ".join([f"{p:.1f}" for p in weather_data['pressure']])
        precip_str = ", ".join([f"{p:.2f}" for p in weather_data['precipitation_probability']])
        
        prompt = f"""Weather data for {weather_data['location']} on {weather_data['datetime']}:
- Temperature (°C): {temp_str}
- Humidity (%): {humidity_str}
- Wind speed (km/h): {wind_str}
- Pressure (hPa): {pressure_str}
- Precipitation probability: {precip_str}

Generate a forecast bulletin:"""
        
        return prompt
    
    def generate_forecast(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_beams: int = 1,
        do_sample: bool = True
    ) -> Tuple[str, float]:
        """
        Generate weather forecast from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            
        Returns:
            Tuple of (forecast_text, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            # Generate with multiple attempts for confidence estimation
            generations = []
            
            for _ in range(3):  # Generate 3 versions
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Decode generated text
                generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                generations.append(generated_text.strip())
            
            # Select best generation (longest reasonable one)
            best_generation = max(generations, key=lambda x: len(x.split()) if len(x.split()) <= 50 else 0)
            
            # Estimate confidence based on generation consistency
            confidence_score = self._estimate_confidence(generations)
            
            return best_generation, confidence_score
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    def _estimate_confidence(self, generations: List[str]) -> float:
        """
        Estimate confidence based on generation consistency.
        
        Args:
            generations: List of generated forecast texts
            
        Returns:
            Confidence score (0-1)
        """
        if len(generations) <= 1:
            return 0.5  # Neutral confidence
        
        # Calculate similarity between generations
        similarities = []
        for i in range(len(generations)):
            for j in range(i + 1, len(generations)):
                # Simple word overlap similarity
                words1 = set(generations[i].lower().split())
                words2 = set(generations[j].lower().split())
                if len(words1.union(words2)) > 0:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    similarities.append(similarity)
        
        # Average similarity as confidence
        if similarities:
            confidence = np.mean(similarities)
            return min(1.0, max(0.1, confidence))  # Clamp to [0.1, 1.0]
        else:
            return 0.5
    
    def process_request(self, request: ForecastRequest) -> ForecastResponse:
        """
        Process a complete forecast request.
        
        Args:
            request: ForecastRequest object
            
        Returns:
            ForecastResponse object
        """
        start_time = datetime.now()
        
        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()
            
            # Fetch current weather data
            weather_data = self.fetch_current_weather(request.location)
            
            # Create prompt
            prompt = self.create_prompt(weather_data)
            
            # Generate forecast
            forecast_text, confidence_score = self.generate_forecast(prompt)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create response
            response = ForecastResponse(
                forecast_text=forecast_text,
                confidence_score=confidence_score if request.include_confidence else None,
                location=request.location.name,
                generated_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=request.forecast_horizon),
                model_version=self.model_version,
                input_data=weather_data,
                processing_time_ms=processing_time
            )
            
            logger.info(f"Generated forecast for {request.location.name} in {processing_time:.0f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error processing forecast request: {e}")
            raise
    
    def batch_process(self, requests: List[ForecastRequest]) -> List[ForecastResponse]:
        """
        Process multiple forecast requests in batch.
        
        Args:
            requests: List of ForecastRequest objects
            
        Returns:
            List of ForecastResponse objects
        """
        logger.info(f"Processing batch of {len(requests)} forecast requests")
        
        responses = []
        for i, request in enumerate(requests):
            try:
                response = self.process_request(request)
                responses.append(response)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(requests)} requests")
                    
            except Exception as e:
                logger.error(f"Error processing request {i}: {e}")
                # Create error response
                error_response = ForecastResponse(
                    forecast_text=f"Error generating forecast: {str(e)}",
                    confidence_score=0.0,
                    location=request.location.name,
                    generated_at=datetime.now(),
                    valid_until=datetime.now(),
                    model_version=self.model_version
                )
                responses.append(error_response)
        
        logger.info(f"Batch processing completed: {len(responses)} responses")
        return responses


class ForecastAPI:
    """
    RESTful API wrapper for weather forecasting inference.
    
    Provides HTTP endpoints for forecast generation with proper
    error handling and response formatting.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize forecast API.
        
        Args:
            model_path: Path to trained model
        """
        self.inference_engine = WeatherInference(model_path)
        self.inference_engine.load_model()
        
        logger.info("ForecastAPI initialized")
    
    async def generate_forecast_endpoint(self, location_data: Dict) -> Dict:
        """
        API endpoint for forecast generation.
        
        Args:
            location_data: Dictionary with location information
            
        Returns:
            JSON response with forecast
        """
        try:
            # Parse location
            location = WeatherLocation(
                name=location_data['name'],
                latitude=location_data['latitude'], 
                longitude=location_data['longitude']
            )
            
            # Create request
            request = ForecastRequest(
                location=location,
                forecast_horizon=location_data.get('forecast_horizon', 6),
                include_confidence=location_data.get('include_confidence', True)
            )
            
            # Process request
            response = self.inference_engine.process_request(request)
            
            # Format response
            return {
                'status': 'success',
                'forecast': response.forecast_text,
                'confidence': response.confidence_score,
                'location': response.location,
                'generated_at': response.generated_at.isoformat(),
                'valid_until': response.valid_until.isoformat(),
                'model_version': response.model_version,
                'processing_time_ms': response.processing_time_ms
            }
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def health_check(self) -> Dict:
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'model_version': self.inference_engine.model_version,
            'model_loaded': self.inference_engine.model is not None,
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Example usage of weather inference."""
    # Example model path (would be actual trained model)
    model_path = "./models/weather-lora-ppo"
    
    # Initialize inference engine
    inference = WeatherInference(model_path)
    
    # Example location
    location = WeatherLocation("New York", 40.7128, -74.0060)
    
    # Create request
    request = ForecastRequest(location=location, forecast_horizon=6)
    
    try:
        # This would work with a trained model
        # response = inference.process_request(request)
        # print(f"Forecast: {response.forecast_text}")
        # print(f"Confidence: {response.confidence_score:.2f}")
        
        logger.info("Example inference setup completed")
        logger.info("To use: train model first, then load for inference")
        
    except Exception as e:
        logger.info(f"Expected error (no trained model): {e}")


if __name__ == "__main__":
    main()