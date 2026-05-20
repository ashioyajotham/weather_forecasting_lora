# Getting Started

This repository currently supports a TinyLlama + PEFT LoRA workflow with
llama.cpp/GGUF inference. The older orchestration commands
`run_complete_pipeline.py` and `train_sft.py` are not present in this checkout.

## Verified Workflow

### 1. Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

The code is PyTorch-first. TensorFlow is intentionally disabled during package
import to avoid optional Keras compatibility issues in Transformers.

### 2. Verify Package And Data

```powershell
python -c "import src; import src.data; print('imports ok')"
python -c "import json; print(len(json.load(open('data/processed/train.json', encoding='utf-8'))))"
```

Expected local data files:

- `data/processed/train.json`
- `data/processed/val.json`
- `data/processed/test.json`

These data files are ignored by git because they can be large. Regenerate them
with:

```powershell
python collect_sample_data.py
```

### 3. Train LoRA Adapter

```powershell
python train_lora_peft.py
```

The current training script uses:

- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Output: `models/weather-lora-peft/lora_adapter`
- Data: `data/processed/train.json`
- CPU-friendly defaults

### 4. Merge Adapter

```powershell
python merge_lora.py
```

This writes a merged Hugging Face model to:

```text
models/weather-merged
```

### 5. Convert To GGUF

After llama.cpp is available locally, convert the merged model:

```powershell
python llama.cpp\convert_hf_to_gguf.py models/weather-merged --outfile models/gguf/weather-tinyllama.gguf --outtype f16
```

The CLI expects:

```text
models/gguf/weather-tinyllama.gguf
llama.cpp/build/bin/Release/llama-server.exe
```

Both are ignored by git and must be produced or restored locally.

### 6. Run CLI

```powershell
python weather_cli.py
```

The CLI starts a local llama.cpp server, waits for `/health`, then sends prompts
to `/completion`.

## Tests And Smoke Checks

```powershell
python -m pytest -q
python scripts/smoke_check.py
```

The smoke check validates imports, processed data shape, config files, model
artifact presence, and CLI prerequisites without loading the model.

## Current Non-Goals

These are not end-to-end runnable in this checkout yet:

- `run_complete_pipeline.py`
- `train_sft.py`
- A runnable FastAPI server entrypoint
- Verified PPO/RLHF training with the currently installed TRL version

