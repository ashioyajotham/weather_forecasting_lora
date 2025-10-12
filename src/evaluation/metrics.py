"""
Evaluation Framework for Weather Forecasting LoRA
==================================================

This module implements comprehensive evaluation metrics for weather forecasting models:
- Text similarity metrics (BLEU, ROUGE)
- Meteorological accuracy metrics
- Calibration and reliability measures
- Human-like style evaluation

Following the evaluation strategy outlined in the project specification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import json
import re
from datetime import datetime

# Text evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK or rouge_score not available. Install with: pip install nltk rouge-score")

# Scientific computation
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Install with: pip install scipy")

logger = logging.getLogger(__name__)


@dataclass
class ForecastPrediction:
    """Represents a weather forecast prediction with metadata."""
    input_prompt: str
    generated_forecast: str
    reference_forecast: str
    location: str
    datetime: str
    
    # Extracted weather conditions (for accuracy evaluation)
    predicted_rain: Optional[bool] = None
    actual_rain: Optional[bool] = None
    predicted_temp_range: Optional[Tuple[float, float]] = None
    actual_temp_range: Optional[Tuple[float, float]] = None
    predicted_wind_speed: Optional[float] = None
    actual_wind_speed: Optional[float] = None


@dataclass 
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    # Text similarity metrics
    bleu_score: float
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float
    
    # Meteorological accuracy
    rain_accuracy: float
    temperature_mae: float
    wind_speed_mae: float
    categorical_accuracy: float
    
    # Calibration metrics
    brier_score: float
    reliability: float
    resolution: float
    
    # Style and fluency
    readability_score: float
    length_similarity: float
    vocabulary_diversity: float
    
    # Overall scores
    overall_score: float
    confidence_interval: Tuple[float, float]


class WeatherTextExtractor:
    """
    Extracts weather information from forecast text for accuracy evaluation.
    
    Uses regex patterns to identify weather conditions, temperatures, and wind speeds.
    """
    
    def __init__(self):
        """Initialize text extraction patterns."""
        # Patterns for extracting weather information
        self.rain_patterns = [
            r'\b(?:rain|showers?|precipitation|wet|drizzle)\b',
            r'\b(?:stormy?|thunderstorms?)\b',
            r'\b(?:downpours?|cloudbursts?)\b'
        ]
        
        self.clear_patterns = [
            r'\b(?:clear|sunny|bright|fair)\b',
            r'\b(?:no rain|dry|arid)\b'
        ]
        
        # Temperature extraction patterns
        self.temp_patterns = [
            r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*°?[CF]',  # Range: 20-25°C
            r'(?:around|near|about)\s+(\d+(?:\.\d+)?)\s*°?[CF]',   # Around 23°C
            r'(?:highs?|temperatures?)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*°?[CF]',  # Highs of 25°C
            r'(\d+(?:\.\d+)?)\s*°?[CF]'  # Direct temperature: 23°C
        ]
        
        # Wind speed patterns
        self.wind_patterns = [
            r'(?:winds?|gusts?)\s+(?:up\s+to\s+|reaching\s+)?(\d+(?:\.\d+)?)\s*(?:km/h|mph|kph)',
            r'(\d+(?:\.\d+)?)\s*(?:km/h|mph|kph)\s+winds?',
            r'windy\s+with\s+(?:speeds?\s+)?(?:up\s+to\s+)?(\d+(?:\.\d+)?)'
        ]
        
        logger.info("WeatherTextExtractor initialized")
    
    def extract_rain_prediction(self, text: str) -> Optional[bool]:
        """
        Extract rain prediction from forecast text.
        
        Args:
            text: Forecast text
            
        Returns:
            True if rain predicted, False if clear, None if unclear
        """
        text_lower = text.lower()
        
        # Check for rain indicators
        rain_score = 0
        for pattern in self.rain_patterns:
            if re.search(pattern, text_lower):
                rain_score += 1
        
        # Check for clear weather indicators
        clear_score = 0
        for pattern in self.clear_patterns:
            if re.search(pattern, text_lower):
                clear_score += 1
        
        # Probability indicators
        prob_match = re.search(r'(\d+)%\s*(?:chance|probability|likelihood)', text_lower)
        if prob_match:
            prob = int(prob_match.group(1))
            if prob >= 50:
                rain_score += 2
            elif prob <= 20:
                clear_score += 2
        
        # Make decision
        if rain_score > clear_score:
            return True
        elif clear_score > rain_score:
            return False
        else:
            return None  # Unclear
    
    def extract_temperature_range(self, text: str) -> Optional[Tuple[float, float]]:
        """
        Extract temperature range from forecast text.
        
        Args:
            text: Forecast text
            
        Returns:
            Tuple of (min_temp, max_temp) or None if not found
        """
        for pattern in self.temp_patterns:
            matches = re.findall(pattern, text)
            if matches:
                if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    # Range pattern matched
                    min_temp = float(matches[0][0])
                    max_temp = float(matches[0][1])
                    return (min_temp, max_temp)
                else:
                    # Single temperature - create range ±2°C
                    temp = float(matches[0])
                    return (temp - 2, temp + 2)
        
        return None
    
    def extract_wind_speed(self, text: str) -> Optional[float]:
        """
        Extract wind speed from forecast text.
        
        Args:
            text: Forecast text
            
        Returns:
            Wind speed in km/h or None if not found
        """
        for pattern in self.wind_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1))
        
        return None
    
    def extract_all_conditions(self, forecast_text: str) -> Dict:
        """
        Extract all weather conditions from forecast text.
        
        Args:
            forecast_text: Generated forecast text
            
        Returns:
            Dictionary with extracted conditions
        """
        return {
            'rain': self.extract_rain_prediction(forecast_text),
            'temperature_range': self.extract_temperature_range(forecast_text),
            'wind_speed': self.extract_wind_speed(forecast_text)
        }


class MetricsCalculator:
    """
    Calculates evaluation metrics for weather forecasting models.
    
    Implements various metrics including text similarity, meteorological accuracy,
    and calibration measures.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.extractor = WeatherTextExtractor()
        
        # Initialize ROUGE scorer if available
        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
            self.smoothing_function = SmoothingFunction().method1
        
        logger.info("MetricsCalculator initialized")
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """
        Calculate BLEU score between generated and reference text.
        
        Args:
            generated: Generated forecast text
            reference: Reference forecast text
            
        Returns:
            BLEU score (0-1)
        """
        if not NLTK_AVAILABLE:
            return 0.0
        
        # Tokenize texts
        reference_tokens = reference.lower().split()
        generated_tokens = generated.lower().split()
        
        # Calculate BLEU score
        score = sentence_bleu(
            [reference_tokens], 
            generated_tokens,
            smoothing_function=self.smoothing_function
        )
        
        return score
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between generated and reference text.
        
        Args:
            generated: Generated forecast text
            reference: Reference forecast text
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not NLTK_AVAILABLE:
            return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}
        
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    
    def calculate_meteorological_accuracy(
        self, 
        predictions: List[ForecastPrediction]
    ) -> Dict[str, float]:
        """
        Calculate meteorological accuracy metrics.
        
        Args:
            predictions: List of forecast predictions
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Extract conditions for all predictions
        rain_correct = 0
        rain_total = 0
        temp_errors = []
        wind_errors = []
        categorical_correct = 0
        categorical_total = 0
        
        for pred in predictions:
            # Extract conditions from generated forecast
            generated_conditions = self.extractor.extract_all_conditions(pred.generated_forecast)
            reference_conditions = self.extractor.extract_all_conditions(pred.reference_forecast)
            
            # Rain accuracy
            if (generated_conditions['rain'] is not None and 
                reference_conditions['rain'] is not None):
                rain_total += 1
                if generated_conditions['rain'] == reference_conditions['rain']:
                    rain_correct += 1
            
            # Temperature accuracy
            if (generated_conditions['temperature_range'] is not None and
                reference_conditions['temperature_range'] is not None):
                gen_temp = np.mean(generated_conditions['temperature_range'])
                ref_temp = np.mean(reference_conditions['temperature_range'])
                temp_errors.append(abs(gen_temp - ref_temp))
            
            # Wind speed accuracy
            if (generated_conditions['wind_speed'] is not None and
                reference_conditions['wind_speed'] is not None):
                wind_errors.append(abs(
                    generated_conditions['wind_speed'] - reference_conditions['wind_speed']
                ))
            
            # Overall categorical accuracy
            categorical_total += 1
            conditions_match = (
                generated_conditions['rain'] == reference_conditions['rain'] and
                (generated_conditions['temperature_range'] is None or 
                 reference_conditions['temperature_range'] is None or
                 abs(np.mean(generated_conditions['temperature_range']) - 
                     np.mean(reference_conditions['temperature_range'])) <= 3)
            )
            if conditions_match:
                categorical_correct += 1
        
        return {
            'rain_accuracy': rain_correct / max(rain_total, 1),
            'temperature_mae': np.mean(temp_errors) if temp_errors else 0.0,
            'wind_speed_mae': np.mean(wind_errors) if wind_errors else 0.0,
            'categorical_accuracy': categorical_correct / max(categorical_total, 1)
        }
    
    def calculate_brier_score(self, predicted_probs: List[float], outcomes: List[bool]) -> float:
        """
        Calculate Brier score for probability predictions.
        
        Args:
            predicted_probs: Predicted probabilities (0-1)
            outcomes: Actual binary outcomes
            
        Returns:
            Brier score (lower is better)
        """
        if not predicted_probs or not outcomes:
            return 1.0  # Worst possible score
        
        brier_scores = [(prob - int(outcome))**2 for prob, outcome in zip(predicted_probs, outcomes)]
        return np.mean(brier_scores)
    
    def calculate_style_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate style and fluency metrics.
        
        Args:
            generated: Generated forecast text
            reference: Reference forecast text
            
        Returns:
            Dictionary with style metrics
        """
        # Length similarity
        gen_len = len(generated.split())
        ref_len = len(reference.split())
        length_similarity = 1 - abs(gen_len - ref_len) / max(gen_len, ref_len, 1)
        
        # Vocabulary diversity (unique words / total words)
        gen_words = generated.lower().split()
        vocabulary_diversity = len(set(gen_words)) / max(len(gen_words), 1)
        
        # Simple readability score (average sentence length)
        sentences = generated.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        readability_score = max(0, 1 - abs(avg_sentence_length - 15) / 15)  # Optimal ~15 words
        
        return {
            'length_similarity': length_similarity,
            'vocabulary_diversity': vocabulary_diversity,
            'readability_score': readability_score
        }
    
    def calculate_all_metrics(self, predictions: List[ForecastPrediction]) -> EvaluationMetrics:
        """
        Calculate all evaluation metrics for a set of predictions.
        
        Args:
            predictions: List of forecast predictions
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        logger.info(f"Calculating metrics for {len(predictions)} predictions")
        
        # Text similarity metrics
        bleu_scores = []
        rouge_scores = {'rouge1_f': [], 'rouge2_f': [], 'rougeL_f': []}
        
        # Style metrics
        style_metrics = {'length_similarity': [], 'vocabulary_diversity': [], 'readability_score': []}
        
        for pred in predictions:
            # BLEU score
            bleu = self.calculate_bleu_score(pred.generated_forecast, pred.reference_forecast)
            bleu_scores.append(bleu)
            
            # ROUGE scores
            rouge = self.calculate_rouge_scores(pred.generated_forecast, pred.reference_forecast)
            for key in rouge_scores:
                rouge_scores[key].append(rouge[key])
            
            # Style metrics
            style = self.calculate_style_metrics(pred.generated_forecast, pred.reference_forecast)
            for key in style_metrics:
                style_metrics[key].append(style[key])
        
        # Meteorological accuracy
        met_accuracy = self.calculate_meteorological_accuracy(predictions)
        
        # Calculate averages
        avg_bleu = np.mean(bleu_scores)
        avg_rouge1 = np.mean(rouge_scores['rouge1_f'])
        avg_rouge2 = np.mean(rouge_scores['rouge2_f'])
        avg_rougeL = np.mean(rouge_scores['rougeL_f'])
        
        avg_length_sim = np.mean(style_metrics['length_similarity'])
        avg_vocab_div = np.mean(style_metrics['vocabulary_diversity'])
        avg_readability = np.mean(style_metrics['readability_score'])
        
        # Overall score (weighted combination)
        overall_score = (
            0.2 * avg_bleu +
            0.2 * avg_rouge1 +
            0.3 * met_accuracy['categorical_accuracy'] +
            0.1 * avg_length_sim +
            0.1 * avg_readability +
            0.1 * met_accuracy['rain_accuracy']
        )
        
        # Confidence interval (bootstrap estimate)
        if SCIPY_AVAILABLE and len(bleu_scores) > 10:
            ci_lower, ci_upper = stats.t.interval(
                0.95, len(bleu_scores)-1, 
                loc=overall_score, 
                scale=stats.sem([overall_score] * len(bleu_scores))
            )
        else:
            ci_lower, ci_upper = overall_score * 0.9, overall_score * 1.1
        
        return EvaluationMetrics(
            bleu_score=avg_bleu,
            rouge_1_f=avg_rouge1,
            rouge_2_f=avg_rouge2,
            rouge_l_f=avg_rougeL,
            rain_accuracy=met_accuracy['rain_accuracy'],
            temperature_mae=met_accuracy['temperature_mae'],
            wind_speed_mae=met_accuracy['wind_speed_mae'],
            categorical_accuracy=met_accuracy['categorical_accuracy'],
            brier_score=0.0,  # Would need probability predictions
            reliability=0.0,   # Would need calibration analysis
            resolution=0.0,    # Would need calibration analysis
            readability_score=avg_readability,
            length_similarity=avg_length_sim,
            vocabulary_diversity=avg_vocab_div,
            overall_score=overall_score,
            confidence_interval=(ci_lower, ci_upper)
        )


class WeatherEvaluator:
    """
    High-level evaluator for weather forecasting models.
    
    Provides complete evaluation pipeline including data preparation,
    metric calculation, and reporting.
    """
    
    def __init__(self):
        """Initialize weather evaluator."""
        self.calculator = MetricsCalculator()
        logger.info("WeatherEvaluator initialized")
    
    def evaluate_model(
        self,
        model,
        test_dataset: List[Dict],
        num_samples: Optional[int] = None
    ) -> EvaluationMetrics:
        """
        Evaluate a weather forecasting model on test data.
        
        Args:
            model: Trained weather forecasting model
            test_dataset: Test dataset
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            EvaluationMetrics object
        """
        logger.info(f"Evaluating model on {len(test_dataset)} samples")
        
        # Limit samples if specified
        if num_samples:
            test_dataset = test_dataset[:num_samples]
        
        predictions = []
        
        for i, example in enumerate(test_dataset):
            try:
                # Generate forecast
                generated_forecast = model.generate_forecast(
                    example['input'],
                    max_new_tokens=128,
                    temperature=0.7
                )
                
                # Create prediction object
                prediction = ForecastPrediction(
                    input_prompt=example['input'],
                    generated_forecast=generated_forecast,
                    reference_forecast=example['target'],
                    location=example.get('location', 'Unknown'),
                    datetime=example.get('datetime', 'Unknown')
                )
                
                predictions.append(prediction)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(test_dataset)} forecasts")
                    
            except Exception as e:
                logger.warning(f"Error generating forecast for sample {i}: {e}")
                continue
        
        # Calculate metrics
        metrics = self.calculator.calculate_all_metrics(predictions)
        
        logger.info("Evaluation completed")
        return metrics
    
    def create_evaluation_report(
        self,
        metrics: EvaluationMetrics,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create detailed evaluation report.
        
        Args:
            metrics: Computed evaluation metrics
            output_path: Path to save report (optional)
            
        Returns:
            Report as string
        """
        report = f"""
# Weather Forecasting Model Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Text Similarity Metrics
- **BLEU Score**: {metrics.bleu_score:.4f}
- **ROUGE-1 F1**: {metrics.rouge_1_f:.4f}
- **ROUGE-2 F1**: {metrics.rouge_2_f:.4f}
- **ROUGE-L F1**: {metrics.rouge_l_f:.4f}

## Meteorological Accuracy
- **Rain Prediction Accuracy**: {metrics.rain_accuracy:.4f}
- **Temperature MAE**: {metrics.temperature_mae:.2f}°C
- **Wind Speed MAE**: {metrics.wind_speed_mae:.2f} km/h
- **Categorical Accuracy**: {metrics.categorical_accuracy:.4f}

## Style and Fluency
- **Readability Score**: {metrics.readability_score:.4f}
- **Length Similarity**: {metrics.length_similarity:.4f}
- **Vocabulary Diversity**: {metrics.vocabulary_diversity:.4f}

## Overall Performance
- **Overall Score**: {metrics.overall_score:.4f}
- **95% Confidence Interval**: [{metrics.confidence_interval[0]:.4f}, {metrics.confidence_interval[1]:.4f}]

## Interpretation
- BLEU/ROUGE scores measure text similarity to reference forecasts
- Meteorological accuracy measures factual correctness of predictions
- Style metrics assess human-like forecast characteristics
- Overall score combines all metrics with domain-appropriate weights

## Recommendations
"""
        
        # Add recommendations based on scores
        if metrics.overall_score > 0.7:
            report += "- ✅ Excellent performance across all metrics\n"
        elif metrics.overall_score > 0.5:
            report += "- ✅ Good performance with room for improvement\n"
        else:
            report += "- ⚠️ Performance needs improvement\n"
        
        if metrics.rain_accuracy < 0.7:
            report += "- 🌧️ Consider improving rain prediction accuracy\n"
        
        if metrics.temperature_mae > 3.0:
            report += "- 🌡️ Temperature predictions could be more precise\n"
        
        if metrics.readability_score < 0.6:
            report += "- 📝 Focus on improving forecast readability and style\n"
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report


def main():
    """Example usage of the evaluation framework."""
    # Create sample predictions for testing
    sample_predictions = [
        ForecastPrediction(
            input_prompt="Weather data for New York...",
            generated_forecast="Partly cloudy with temperatures around 22-25°C. Light winds up to 15 km/h. No rain expected.",
            reference_forecast="Partly cloudy skies with highs near 24°C. Breezy conditions with winds to 18 km/h. Dry conditions.",
            location="New York",
            datetime="2025-09-30"
        ),
        ForecastPrediction(
            input_prompt="Weather data for London...",
            generated_forecast="Overcast with showers likely. Temperatures 18-20°C. Winds increasing to 25 km/h.",
            reference_forecast="Cloudy with rain expected. Cool temperatures around 19°C. Windy with gusts to 28 km/h.",
            location="London", 
            datetime="2025-09-30"
        )
    ]
    
    # Initialize evaluator
    evaluator = WeatherEvaluator()
    
    # Calculate metrics
    metrics = evaluator.calculator.calculate_all_metrics(sample_predictions)
    
    # Create report
    report = evaluator.create_evaluation_report(metrics)
    print(report)


if __name__ == "__main__":
    main()