# Project Status

Status: usable research prototype, not production-ready.

## Verified Locally

- Package imports work after restoring the `src.data.preprocessor` import path.
- Unit tests pass: `107 passed, 9 skipped`.
- Processed datasets are present locally:
  - train: 13,913 samples
  - validation: 1,739 samples
  - test: 1,740 samples
- TinyLlama LoRA training script exists: `train_lora_peft.py`.
- LoRA merge script exists: `merge_lora.py`.
- CLI exists: `weather_cli.py`.
- Local model artifacts appear under `models/`, but this directory is ignored by git.

## Implemented Components

- Data collection from Open-Meteo in `src/data/collector.py`.
- Weather preprocessing implementation in `src/data/preprocessor.py`.
- TinyLlama PEFT training path in `train_lora_peft.py`.
- LoRA adapter merge path in `merge_lora.py`.
- Evaluation metric utilities in `src/evaluation/metrics.py`.
- llama.cpp-backed terminal CLI in `weather_cli.py`.
- PPO reward/trainer scaffolding in `src/rl/ppo_trainer.py`.

## Known Gaps

- `run_complete_pipeline.py`, `train_sft.py`, and `config/base_config.yaml` are
  referenced by older docs but do not exist.
- PPO/RLHF is scaffolding only until verified against the installed TRL API and
  a real training run.
- FastAPI is only represented by wrapper code, not a runnable app entrypoint.
- The main runnable training script now targets attention and MLP projection
  modules, but existing local GGUF artifacts must be retrained and reconverted
  before generation quality reflects the fix.
- Existing local GGUF output can still repeat prompt/table fragments because
  earlier training supervised the full prompt plus answer instead of masking
  prompt tokens. `train_lora_peft.py` now masks prompt labels for future runs.
- The tests include many placeholders and skipped integration paths, so passing
  tests should not be interpreted as full model-quality validation.
- Data and model artifacts are ignored by git and need explicit regeneration or
  restoration instructions for fresh clones.

## Recommended Next Work

1. Remove or adopt the duplicate untracked `src/preprocessor.py` file after
   confirming it is no longer needed.
2. Keep `scripts/smoke_check.py` current as fresh-checkout validation
   explicit.
3. Align `train_lora_peft.py` target modules with the stated LoRA methodology,
   or update the methodology claim.
4. Add a runnable FastAPI entrypoint if API deployment is still a goal.
5. Update PPO code to the installed TRL version before advertising PPO support.
