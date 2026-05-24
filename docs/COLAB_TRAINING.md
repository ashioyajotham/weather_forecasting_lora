# Colab Training Notes

Use two notebooks, in order:

1. `notebooks/colab_tinyllama_gguf.ipynb`
2. `notebooks/colab_ppo_training.ipynb`

This split is intentional. TinyLlama LoRA training, merge, and GGUF conversion
produce the baseline artifact. PPO is experimental and should consume that
baseline as input instead of being mixed into the same runtime.

## GPU Choice

A Colab T4 can work for TinyLlama LoRA because this project trains LoRA
adapters on TinyLlama, not full model weights. It is acceptable for smoke and
medium runs with conservative settings:

```bash
WEATHER_LORA_MAX_SAMPLES=1000
WEATHER_LORA_REPORT_TO=none
```

T4 is not optimal for full-dataset runs or PPO. Prefer L4, A10, A100, or a
similar larger CUDA GPU when runtime and memory matter.

## Artifact Handoff

The notebooks use Google Drive as the handoff layer:

```text
MyDrive/weather_forecasting_lora_runs/
  tinyllama_gguf_<timestamp>/
    adapter/
    merged/
    gguf/
    manifest_tinyllama_gguf.json
    *.log
  ppo_<timestamp>/
    ppo_model/
    manifest_ppo.json
    *.log
```

Do not push these binaries to git. Import them locally by copying:

```text
adapter/ -> models/weather-lora-peft/lora_adapter/
merged/  -> models/weather-merged/
gguf/weather-tinyllama.gguf -> models/gguf/weather-tinyllama.gguf
```

Then verify locally:

```powershell
python scripts/smoke_check.py
python scripts/eval_smoke.py
python scripts/ppo_smoke.py
python scripts/generation_quality_eval.py
python weather_cli.py
```

For API verification:

```powershell
$env:WEATHER_LLAMA_SERVER_URL = "http://127.0.0.1:8080"
python run_api.py
```

## TinyLlama/GGUF Acceptance

Accept a Colab TinyLlama/GGUF run only when:

- smoke checks pass before training
- LoRA adapter is saved
- merged Hugging Face model is saved
- GGUF conversion completes or the merged model is exported for local conversion
- `manifest_tinyllama_gguf.json` records commit, GPU, sample count, paths, and sizes
- local import passes smoke checks and generation-quality eval

## PPO Acceptance

Run PPO only after a TinyLlama/GGUF Drive run exists. In the PPO notebook, set
`SOURCE_RUN_DIR` to that exact Drive path.

Validate first:

```bash
python train_ppo.py
```

Then start a small CUDA trial:

```bash
python train_ppo.py --run-training --limit 64
```

Treat PPO as experimental until a real CUDA run completes and the imported
result improves reward behavior without degrading pinned generation quality.
