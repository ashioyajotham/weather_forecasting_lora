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
- A 200-sample smoke retrain completed locally with response-only label masking.
- The adapter was merged and reconverted to
  `models/gguf/weather-tinyllama.gguf` as an F16 GGUF artifact.
- llama.cpp smoke inference through the CLI prompt path returns concise
  natural-language forecasts instead of prompt/table fragments.

## Implemented Components

- Data collection from Open-Meteo in `src/data/collector.py`.
- Weather preprocessing implementation in `src/data/preprocessor.py`.
- TinyLlama PEFT training path in `train_lora_peft.py`.
- LoRA adapter merge path in `merge_lora.py`.
- Evaluation metric utilities in `src/evaluation/metrics.py`.
- llama.cpp-backed terminal CLI in `weather_cli.py`.
- PPO reward/trainer scaffolding in `src/rl/ppo_trainer.py`.
- PPO smoke diagnostics in `scripts/ppo_smoke.py`.

## Known Gaps

- `run_complete_pipeline.py`, `train_sft.py`, and `config/base_config.yaml` are
  referenced by older docs but do not exist.
- PPO/RLHF is scaffolding only until verified against the installed TRL API and
  a real training run.
- FastAPI is only represented by wrapper code, not a runnable app entrypoint.
- The latest local retrain was intentionally small for turnaround time. It
  verifies the training/merge/GGUF plumbing, but it is not a full quality run.
- Optional metric dependencies such as NLTK and `rouge-score` are not required
  for smoke validation because fallback metrics are implemented, but installing
  them would improve parity with standard BLEU/ROUGE reporting.
- The tests include many placeholders and skipped integration paths, so passing
  tests should not be interpreted as full model-quality validation.
- Data and model artifacts are ignored by git and need explicit regeneration or
  restoration instructions for fresh clones.
- Local model/log/cache directories are runtime artifacts and should stay out
  of git unless the project adds an explicit release artifact policy.

## Recommended Next Work

1. Run a full-scale retrain on the complete processed dataset, then repeat the
   merge, GGUF conversion, CLI smoke test, and quantitative evaluation.
2. Add a small pinned generation-quality eval set so future GGUF outputs can be
   compared against expected forecast style and content.
3. Keep `scripts/smoke_check.py`, `scripts/eval_smoke.py`, and
   `scripts/ppo_smoke.py` current as fresh-checkout validation.
4. Add a runnable FastAPI entrypoint if API deployment is still a goal.
5. Run and document real PPO/RLHF training before advertising PPO support
   beyond reward-model and import smoke diagnostics.
