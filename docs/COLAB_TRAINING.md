# Colab Training Notes

A Colab T4 can work for this project because the active model path is
TinyLlama plus LoRA adapters, not full-parameter fine-tuning. It is a practical
baseline for smoke and medium runs, but it is not optimal for long full-dataset
experiments.

Recommended T4 starting point:

```bash
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export WANDB_DISABLED=true
export WEATHER_LORA_REPORT_TO=none
export WEATHER_LORA_MAX_SAMPLES=1000
python train_lora_peft.py
python merge_lora.py
python scripts/generation_quality_eval.py
```

Use smaller sample counts first if memory is tight. Increase toward the full
processed dataset only after a clean smoke run.

PPO needs CUDA. The local CPU/offload path is expected to reject PPO training
because TRL value-head models do not support CPU or disk offload. On Colab,
validate first:

```bash
python train_ppo.py
```

Then start a small PPO run:

```bash
python train_ppo.py --run-training --limit 64
```

For better throughput or larger PPO batches, prefer L4, A10, A100, or similar
GPUs over T4.
