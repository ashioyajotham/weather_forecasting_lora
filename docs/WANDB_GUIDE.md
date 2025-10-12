# Weights & Biases Integration Guide
## Weather Forecasting LoRA Project

This guide explains how to use Weights & Biases (W&B) for experiment tracking in the Weather Forecasting LoRA project.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Configuration](#configuration)
4. [Training with W&B](#training-with-wb)
5. [Metrics Tracked](#metrics-tracked)
6. [W&B Dashboard](#wb-dashboard)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

W&B integration provides:
- **Automatic experiment tracking** - All training metrics logged automatically
- **Model versioning** - Save and version model checkpoints as artifacts
- **Visualization** - Interactive charts for loss curves, metrics, and predictions
- **Collaboration** - Share experiments with team members
- **Reproducibility** - Track configurations and code versions

### What Gets Tracked?

✅ Training metrics (loss, learning rate, gradient norms)  
✅ Evaluation metrics (BLEU, ROUGE, meteorological accuracy)  
✅ Model hyperparameters and configurations  
✅ Sample predictions and comparisons  
✅ Model checkpoints as artifacts  
✅ System metrics (GPU utilization, memory)  
✅ Code snapshots for reproducibility  

---

## Setup

### 1. Install W&B

```bash
pip install wandb
```

Already installed in this project! ✅

### 2. Login to W&B

First-time setup:

```bash
wandb login
```

This will open your browser. Copy your API key from https://wandb.ai/authorize and paste it in the terminal.

### 3. Set Your Entity (Optional)

Update `config/base_config.yaml`:

```yaml
wandb:
  entity: "your-username-or-team"  # Change from null
```

Or set via environment variable:

```bash
export WANDB_ENTITY="your-username"
```

---

## Configuration

### Base Configuration File

The W&B settings are in `config/base_config.yaml`:

```yaml
wandb:
  # Project settings
  project: "weather-forecasting-lora"      # W&B project name
  entity: null                              # Your username/team (or null for default)
  name: null                                # Run name (auto-generated if null)
  group: "sft-experiments"                  # Group related experiments
  tags: ["lora", "weather-forecasting", "schulman-2025"]
  notes: "Weather forecasting with LoRA following Schulman et al. (2025)"
  
  # Logging configuration
  log_model: "checkpoint"                   # Log model checkpoints
  log_freq: 100                             # Log metrics every 100 steps
  log_gradients: true                       # Track gradient statistics
  log_parameters: true                      # Track parameter statistics
  
  # Model watching
  watch_model: true                         # Watch model architecture
  watch_freq: 1000                          # Gradient/parameter log frequency
  
  # Artifacts
  save_code: true                           # Save code snapshot
  log_artifacts: true                       # Save model checkpoints as artifacts
  
  # Evaluation tracking
  log_predictions: true                     # Log sample predictions
  num_predictions: 10                       # Number of predictions to log
  
  # Custom metrics tracked
  custom_metrics:
    - "bleu_score"
    - "rouge_1_f"
    - "rouge_2_f"
    - "rouge_l_f"
    - "temperature_mae"
    - "temperature_accuracy"
    - "wind_speed_mae"
    - "precipitation_accuracy"
  
  # System metrics
  log_system_metrics: true                  # Track GPU/CPU/memory
```

### Runtime Configuration Overrides

You can override config settings via command-line arguments:

```bash
python train_lora.py \
  --wandb_project "my-custom-project" \
  --wandb_run_name "experiment-v1"
```

Or disable W&B entirely:

```bash
python train_lora.py --no_wandb
```

---

## Training with W&B

### Basic Training

Simply run the training script - W&B is enabled by default:

```bash
python train_lora.py \
  --train_data data/processed/train.json \
  --val_data data/processed/val.json \
  --test_data data/processed/test.json \
  --output_dir models/weather-lora-v1
```

### Custom Experiment Name

```bash
python train_lora.py \
  --wandb_run_name "schulman-lora-r32-lr5e-5" \
  --output_dir models/experiment-1
```

### Training Without W&B

```bash
python train_lora.py --no_wandb
```

### What Happens During Training?

1. **Initialization**: W&B run is created with your config
2. **Dashboard Link**: URL printed to console (e.g., `https://wandb.ai/your-user/project/runs/abc123`)
3. **Auto-Logging**: Metrics logged every `log_freq` steps
4. **Checkpoint Saving**: Model artifacts uploaded on save
5. **Finalization**: Run marked as complete when training finishes

---

## Metrics Tracked

### Training Metrics

| Metric | Description | Namespace |
|--------|-------------|-----------|
| `train/loss` | Training loss | `train/` |
| `train/learning_rate` | Current learning rate | `train/` |
| `train/epoch` | Current epoch | `train/` |
| `train/grad_norm` | Gradient norm | `train/` |

### NLG Metrics (Text Quality)

| Metric | Description | Namespace |
|--------|-------------|-----------|
| `nlg/bleu_score` | BLEU score (0-1) | `nlg/` |
| `nlg/rouge_1_f` | ROUGE-1 F1 score | `nlg/` |
| `nlg/rouge_2_f` | ROUGE-2 F1 score | `nlg/` |
| `nlg/rouge_l_f` | ROUGE-L F1 score | `nlg/` |

### Weather-Specific Metrics

| Metric | Description | Namespace |
|--------|-------------|-----------|
| `weather/temperature_mae` | Temperature MAE (°C) | `weather/` |
| `weather/temperature_accuracy` | Temperature accuracy (0-1) | `weather/` |
| `weather/wind_speed_mae` | Wind speed MAE (km/h) | `weather/` |
| `weather/precipitation_accuracy` | Precipitation accuracy (0-1) | `weather/` |

### Evaluation Metrics

| Metric | Description | Namespace |
|--------|-------------|-----------|
| `eval/readability_score` | Text readability | `eval/` |
| `eval/length_similarity` | Length similarity to reference | `eval/` |
| `eval/vocabulary_diversity` | Vocabulary richness | `eval/` |
| `eval/overall_score` | Combined performance score | `eval/` |

### System Metrics (Automatic)

- GPU utilization (%)
- GPU memory usage (MB)
- CPU utilization (%)
- System memory usage (MB)
- Process CPU/memory

---

## W&B Dashboard

### Accessing Your Dashboard

After starting training, you'll see:

```
✅ W&B initialized: schulman-lora-r32-lr5e-5 (abc123xyz)
📊 Dashboard: https://wandb.ai/your-user/weather-forecasting-lora/runs/abc123xyz
```

Click the link to open your live dashboard!

### Dashboard Sections

#### 1. **Overview Tab**
- Run metadata (name, tags, notes)
- Configuration summary
- Training duration
- Quick metrics summary

#### 2. **Charts Tab**
- **Loss Curves**: Training and validation loss over time
- **Learning Rate Schedule**: LR changes during training
- **NLG Metrics**: BLEU, ROUGE scores
- **Weather Metrics**: Temperature MAE, wind speed MAE, etc.
- **System Metrics**: GPU utilization, memory usage

#### 3. **System Tab**
- Hardware utilization graphs
- CPU/GPU/memory trends
- Process statistics

#### 4. **Logs Tab**
- Training logs in real-time
- Console output

#### 5. **Files Tab**
- Configuration files
- Output logs
- Model checkpoints (if saved)

#### 6. **Artifacts Tab**
- Model checkpoints with versions
- Download trained models
- Compare artifact versions

#### 7. **Predictions Table**
- Sample input prompts
- Model predictions
- Reference forecasts
- Side-by-side comparisons

---

## Advanced Features

### 1. Programmatic W&B Usage

For custom scripts or notebooks:

```python
from src.utils.wandb_logger import WandBLogger

# Initialize logger
logger = WandBLogger(config_path="config/base_config.yaml")

# Start run
logger.init(
    run_name="my-experiment",
    config={"lr": 5e-5, "epochs": 3}
)

# Log metrics
logger.log_metrics({
    "custom_metric": 0.85,
    "another_metric": 123
}, step=0)

# Log training metrics
logger.log_training_metrics(
    loss=0.42,
    learning_rate=5e-5,
    epoch=1,
    step=100,
    grad_norm=0.8
)

# Log model artifact
logger.log_model_artifact(
    model_path="models/checkpoint-1000",
    artifact_name="checkpoint-1000",
    aliases=["latest", "best"]
)

# Finish run
logger.finish()
```

### 2. Context Manager Usage

```python
with WandBLogger(config_path="config/base_config.yaml") as logger:
    logger.init(run_name="experiment-1")
    
    # Training loop
    for epoch in range(3):
        logger.log_metrics({"epoch": epoch}, step=epoch)
    
    # Auto-finishes on exit
```

### 3. Custom Visualizations

```python
import wandb

# Log custom plot
wandb.log({
    "custom_plot": wandb.plot.scatter(
        wandb.Table(data=[[1, 2], [3, 4]], columns=["x", "y"]),
        "x", "y"
    )
})

# Log images
wandb.log({"prediction_viz": wandb.Image("path/to/image.png")})

# Log histograms
wandb.log({"distribution": wandb.Histogram(np_array)})
```

### 4. Comparing Experiments

In the W&B web interface:
1. Go to your project page
2. Select multiple runs (checkbox on left)
3. Click "Compare" button
4. View side-by-side metrics, configs, and charts

### 5. Hyperparameter Sweeps

Create a sweep configuration (`sweep_config.yaml`):

```yaml
program: train_lora.py
method: bayes
metric:
  name: eval/overall_score
  goal: maximize
parameters:
  learning_rate:
    min: 1e-5
    max: 1e-4
  lora.r:
    values: [16, 32, 64]
  lora.alpha:
    values: [16, 32, 64]
```

Run sweep:

```bash
# Initialize sweep
wandb sweep sweep_config.yaml

# Run sweep agents
wandb agent your-username/weather-forecasting-lora/sweep-id
```

---

## Best Practices

### 🏷️ Naming Conventions

**Good run names:**
- `schulman-lora-r32-a32-lr5e5` (descriptive)
- `exp-001-baseline` (numbered experiments)
- `ablation-no-mlp-adapters` (ablation studies)

**Bad run names:**
- `test` (not descriptive)
- `asdf` (meaningless)
- `final-v2-FINAL-FINAL` (version chaos)

### 📊 Organizing Experiments

**Use Groups:**
```python
logger.init(run_name="exp-1", config={"group": "baseline-experiments"})
logger.init(run_name="exp-2", config={"group": "ablation-studies"})
```

**Use Tags:**
```python
tags = ["lora", "r=32", "quantized", "schulman-2025"]
logger.init(run_name="exp-1", config={"tags": tags})
```

### 💾 Artifact Versioning

**Alias conventions:**
- `latest`: Most recent checkpoint
- `best`: Best performing checkpoint
- `final`: Final trained model
- `v1`, `v2`: Version numbers

```python
logger.log_model_artifact(
    model_path="models/checkpoint",
    artifact_name="weather-lora",
    aliases=["latest", "best", "v1.0"]
)
```

### 📝 Documentation

**Add notes to runs:**
```python
logger.init(
    run_name="exp-1",
    config={
        "notes": "Testing Schulman methodology with r=32, alpha=32. "
                 "Hypothesis: Higher rank improves meteorological accuracy."
    }
)
```

### 🔒 Privacy & Teams

**Public vs Private:**
- Personal account: Runs are private by default
- Team account: Configure privacy settings

**Sharing:**
- Share run URL directly
- Add team members to project
- Create reports for stakeholders

---

## Troubleshooting

### Issue: "wandb: ERROR Unable to authenticate"

**Solution:**
```bash
wandb login --relogin
```

### Issue: "wandb: WARNING Run initialization failed"

**Solution:**
```bash
# Check internet connection
ping wandb.ai

# Verify API key
wandb login --verify

# Check proxy settings if behind firewall
export WANDB_HTTP_PROXY="http://proxy:port"
```

### Issue: Runs not showing up in dashboard

**Solution:**
- Check you're logged into correct account
- Verify project name matches config
- Wait 10-30 seconds for sync
- Check W&B status: https://status.wandb.ai

### Issue: Disk space filling up with artifacts

**Solution:**
```yaml
# In config/base_config.yaml
wandb:
  log_artifacts: false  # Disable artifact logging
  log_model: false      # Don't save model checkpoints
```

Or clean up old artifacts:
```bash
wandb artifact cache cleanup
```

### Issue: Training slowing down due to W&B

**Solution:**
```yaml
# Reduce logging frequency
wandb:
  log_freq: 500         # Increase from 100
  watch_freq: 5000      # Increase from 1000
  log_predictions: false # Disable if not needed
```

### Issue: Offline training (no internet)

**Solution:**
```bash
# Run in offline mode
export WANDB_MODE=offline

python train_lora.py

# Later, sync runs when online
wandb sync runs/offline-run-xyz
```

---

## Additional Resources

### Official Documentation
- **W&B Docs**: https://docs.wandb.ai
- **Python Library**: https://docs.wandb.ai/ref/python
- **Examples**: https://github.com/wandb/examples

### Project-Specific
- **Configuration**: `config/base_config.yaml`
- **W&B Logger**: `src/utils/wandb_logger.py`
- **Training Script**: `train_lora.py`
- **Evaluation**: `src/evaluation/metrics.py`

### Support
- **W&B Community**: https://community.wandb.ai
- **GitHub Issues**: https://github.com/wandb/wandb/issues
- **Email Support**: support@wandb.com

---

## Quick Reference

### Common Commands

```bash
# Login
wandb login

# Check status
wandb status

# List projects
wandb projects

# List runs
wandb runs weather-forecasting-lora

# Pull artifact
wandb artifact get username/project/artifact:version

# Sync offline runs
wandb sync runs/offline-run-xyz

# Clean cache
wandb artifact cache cleanup
```

### Environment Variables

```bash
# Disable W&B
export WANDB_DISABLED=true

# Offline mode
export WANDB_MODE=offline

# Set entity
export WANDB_ENTITY="your-username"

# Set project
export WANDB_PROJECT="weather-forecasting-lora"

# API key
export WANDB_API_KEY="your-api-key"

# HTTP proxy
export WANDB_HTTP_PROXY="http://proxy:8080"
```

---

## Conclusion

W&B integration in this project provides:
✅ **Automatic tracking** of all experiments  
✅ **Reproducible** results with config versioning  
✅ **Collaborative** workspace for team research  
✅ **Visual** insights into model performance  
✅ **Versioned** model artifacts for deployment  

Following Schulman et al. (2025) methodology with proper experiment tracking! 🚀

For questions or issues, check the [Troubleshooting](#troubleshooting) section or refer to the [official W&B documentation](https://docs.wandb.ai).

---

**Happy Experimenting! 🎉**
