# Colab Training Notes

Use two notebooks, in order:

1. `notebooks/colab_tinyllama_gguf.ipynb`
2. `notebooks/colab_ppo_training.ipynb`

This split is intentional. TinyLlama LoRA training, merge, and GGUF conversion
produce the baseline artifact. PPO is experimental and should consume that
baseline as input instead of being mixed into the same runtime.

## Fresh Runtime Rule

If a Colab run fails during dependency installation or import preflight, restart
the runtime before trying again. Do not continue after package mismatch errors.

The known failure mode is installing the full local `requirements.txt` on Colab:
it can replace only part of Colab's CUDA PyTorch stack and break Transformers
imports with `operator torchvision::nms does not exist`. The notebooks therefore
install `requirements-colab.txt`, which leaves Colab's `torch`, `torchvision`,
and `torchaudio` packages intact.

## GPU Choice

A Colab T4 can work for TinyLlama LoRA because this project trains LoRA
adapters on TinyLlama, not full model weights. It is acceptable for smoke and
medium runs with conservative settings:

```bash
WEATHER_LORA_MAX_SAMPLES=200
WEATHER_LORA_REPORT_TO=none
```

Use `200` for the first retry, then increase to `1000` or full data after the
adapter, merge, GGUF conversion, and manifest export pass. T4 is not optimal
for full-dataset runs or PPO. Prefer L4, A10, A100, or a similar larger CUDA GPU
when runtime and memory matter.

## Processed Data On Colab

Fresh clones do not include `data/processed/train.json`, `val.json`, or
`test.json` because those files are ignored project artifacts. The TinyLlama
notebook runs `scripts/prepare_data.py` before smoke checks:

```bash
python scripts/prepare_data.py --min-train 240 --min-val 30 --min-test 30 --synthetic-if-missing
```

If real processed data is already present, the script validates and uses it. If
not, it creates a deterministic synthetic fallback dataset for a 200-sample
smoke run. This fallback verifies the training, merge, GGUF, and artifact
plumbing. It is not a substitute for final-quality model training.

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

- dependency preflight imports `Trainer`, `PeftModel`, and CUDA Torch
- processed data is validated or generated before smoke checks
- smoke checks pass before training
- LoRA adapter is saved
- merged Hugging Face model is saved
- GGUF conversion completes in the isolated conversion venv, or the merged model
  is exported for local conversion
- `manifest_tinyllama_gguf.json` records commit, GPU, sample count, paths, and
  nonzero artifact sizes
- local import passes smoke checks and generation-quality eval

For full-quality training, restore or regenerate the larger real processed
dataset in `data/processed/` before increasing `WEATHER_LORA_MAX_SAMPLES`
beyond smoke scale.

## PPO Acceptance

Run PPO only after a TinyLlama/GGUF Drive run exists. The PPO notebook now
auto-detects the latest valid `tinyllama_gguf_*` Drive run by default. Set the
`SOURCE_RUN_DIR` environment variable only when you want to force a specific
run.

The selected source run must include:

- `manifest_tinyllama_gguf.json`
- `merged/`
- Hugging Face model weights inside `merged/`, such as `.safetensors`, `.bin`,
  or a model weight index file

Fresh Colab clones also lack ignored `data/processed/*.json` files. Before PPO
smoke checks, the notebook runs:

```bash
python scripts/prepare_data.py --min-train 64 --min-val 8 --min-test 8 --synthetic-if-missing
```

This keeps PPO plumbing reproducible. For final PPO quality, restore or
regenerate the real processed dataset before increasing sample count.

Validate first:

```bash
python train_ppo.py
```

Then start a small CUDA trial:

```bash
python train_ppo.py --run-training --limit 16
```

Treat PPO as experimental until a real CUDA run completes and the imported
result improves reward behavior without degrading pinned generation quality.
