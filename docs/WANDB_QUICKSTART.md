# W&B Quick Start Guide
## Get Started in 5 Minutes! ⚡

This is a quick reference for using W&B in the Weather Forecasting LoRA project.

---

## 🚀 Setup (One-Time)

```bash
# 1. Login to W&B
wandb login

# 2. (Optional) Set your username/team
# Edit config/base_config.yaml and set:
# wandb:
#   entity: "your-username"
```

---

## 🎯 Training Commands

### Basic Training
```bash
python train_lora.py \
  --train_data data/processed/train.json \
  --val_data data/processed/val.json \
  --test_data data/processed/test.json
```

### Named Experiment
```bash
python train_lora.py \
  --wandb_run_name "schulman-lora-baseline" \
  --output_dir models/baseline
```

### Custom Project
```bash
python train_lora.py \
  --wandb_project "my-weather-experiments" \
  --wandb_run_name "exp-001"
```

### Without W&B
```bash
python train_lora.py --no_wandb
```

---

## 📊 What You'll See

After starting training, you'll get:

```
✅ W&B initialized: schulman-lora-baseline (abc123)
📊 Dashboard: https://wandb.ai/username/weather-forecasting-lora/runs/abc123
🚀 Starting LoRA fine-tuning...
```

Click the dashboard link to see:
- 📈 Real-time loss curves
- 🎯 Evaluation metrics (BLEU, ROUGE, temperature MAE, etc.)
- 💾 Model checkpoints
- 📝 Sample predictions
- ⚙️ GPU/CPU/memory usage

---

## 🔍 Metrics Logged

### Training (every 100 steps)
- `train/loss` - Training loss
- `train/learning_rate` - Current LR
- `train/grad_norm` - Gradient norm

### Evaluation (every eval_steps)
- `nlg/bleu_score` - BLEU score
- `nlg/rouge_1_f`, `rouge_2_f`, `rouge_l_f` - ROUGE scores
- `weather/temperature_mae` - Temperature error (°C)
- `weather/wind_speed_mae` - Wind speed error (km/h)
- `weather/precipitation_accuracy` - Rain prediction accuracy
- `eval/overall_score` - Combined performance

### Predictions
- Sample input prompts
- Model forecasts
- Reference forecasts
- Side-by-side comparison

---

## 💾 Model Artifacts

Checkpoints are automatically logged as W&B artifacts:

- `checkpoint-1000`, `checkpoint-2000`, etc. (during training)
- `weather-lora-final` (final model with aliases: "final", "best")

Download a model:
```bash
wandb artifact get username/weather-forecasting-lora/weather-lora-final:latest
```

---

## 🔧 Configuration

All W&B settings are in `config/base_config.yaml`:

```yaml
wandb:
  project: "weather-forecasting-lora"  # Your project name
  entity: null                          # Your username (or null)
  group: "sft-experiments"              # Group related runs
  tags: ["lora", "weather", "schulman-2025"]
  log_freq: 100                         # Log every 100 steps
  log_predictions: true                 # Log sample predictions
```

Override via CLI:
```bash
python train_lora.py --wandb_project "new-project" --wandb_run_name "test-1"
```

---

## 📚 Full Documentation

For detailed information, see:
- **Full Guide:** `docs/WANDB_GUIDE.md`
- **Integration Summary:** `docs/WANDB_INTEGRATION_SUMMARY.md`
- **W&B Logger Code:** `src/utils/wandb_logger.py`
- **Training Script:** `train_lora.py`

---

## 🆘 Quick Troubleshooting

**Can't login?**
```bash
wandb login --relogin
```

**Runs not showing?**
- Check internet connection
- Wait 10-30 seconds for sync
- Verify project name matches config

**Offline training?**
```bash
export WANDB_MODE=offline
python train_lora.py
# Later: wandb sync runs/offline-run-xyz
```

**Disable W&B?**
```bash
python train_lora.py --no_wandb
```

---

## ✨ Example Workflow

```bash
# 1. Login
wandb login

# 2. Train baseline
python train_lora.py \
  --wandb_run_name "baseline-r32-lr5e5" \
  --output_dir models/baseline

# 3. Train ablation (different rank)
python train_lora.py \
  --wandb_run_name "ablation-r64-lr5e5" \
  --output_dir models/ablation-r64

# 4. Compare experiments in W&B dashboard
# Select both runs → Click "Compare"

# 5. Evaluate best model
python train_lora.py \
  --eval_only \
  --model_path models/baseline \
  --test_data data/processed/test.json
```

---

## 🎓 Best Practices

### Naming
✅ `schulman-lora-r32-lr5e5` (descriptive)  
✅ `exp-001-baseline` (numbered)  
✅ `ablation-no-mlp` (clear purpose)  
❌ `test`, `asdf`, `final-FINAL` (bad)

### Organization
- Use **groups** for related experiments
- Use **tags** for easy filtering
- Add **notes** explaining hypothesis
- Use **aliases** for important checkpoints

### Sharing
- Click "Share" in dashboard for URL
- Create reports for stakeholders
- Add team members to project

---

## 🚀 You're Ready!

Start tracking your experiments now:

```bash
wandb login
python train_lora.py --wandb_run_name "my-first-experiment"
```

Happy experimenting! 🎉

For more details, check `docs/WANDB_GUIDE.md`
