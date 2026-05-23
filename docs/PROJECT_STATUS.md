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
- FastAPI health and forecast fallback endpoints pass local TestClient smoke
  checks.
- `nltk` and `rouge_score` are installed in the current user Python
  environment; fallback metrics remain available for fresh environments.

## Implemented Components

- Data collection from Open-Meteo in `src/data/collector.py`.
- Weather preprocessing implementation in `src/data/preprocessor.py`.
- TinyLlama PEFT training path in `train_lora_peft.py`.
- LoRA adapter merge path in `merge_lora.py`.
- Evaluation metric utilities in `src/evaluation/metrics.py`.
- llama.cpp-backed terminal CLI in `weather_cli.py`.
- FastAPI app factory in `src/inference/api.py` and runner in `run_api.py`.
- PPO reward/trainer scaffolding in `src/rl/ppo_trainer.py`.
- PPO smoke diagnostics in `scripts/ppo_smoke.py`.
- Pinned generation-quality fixture in `data/eval/generation_quality.json`.

## Known Gaps

- `run_complete_pipeline.py`, `train_sft.py`, and `config/base_config.yaml` are
  referenced by older docs but do not exist.
- PPO/RLHF has reward-model and TRL smoke coverage plus a guarded `train_ppo.py`
  entrypoint, but still needs a real long-running PPO job before being treated
  as a trained model stage.
- FastAPI has a runnable app entrypoint. It supports fallback smoke responses
  and llama.cpp proxying through `WEATHER_LLAMA_SERVER_URL`; production auth,
  rate limiting, and deployment packaging are still open.
- The latest local retrain was intentionally small for turnaround time. It
  verifies the training/merge/GGUF plumbing, but it is not a full quality run.
- Optional metric dependencies such as NLTK and `rouge-score` are pinned and
  installed locally, but fallback metrics remain necessary for fresh or offline
  environments.
- The tests include many placeholders and skipped integration paths, so passing
  tests should not be interpreted as full model-quality validation.
- Data and model artifacts are ignored by git and need explicit regeneration or
  restoration instructions for fresh clones.
- Local model/log/cache directories are runtime artifacts and should stay out
  of git unless the project adds an explicit release artifact policy.

## Recommended Next Work

1. Run a full-scale retrain on the complete processed dataset, then repeat the
   merge, GGUF conversion, CLI smoke test, and quantitative evaluation.
2. Run `scripts/generation_quality_eval.py` against real GGUF outputs after
   each retrain and track the report with the training notes.
3. Keep `scripts/smoke_check.py`, `scripts/eval_smoke.py`,
   `scripts/ppo_smoke.py`, and `scripts/generation_quality_eval.py` current as
   fresh-checkout validation.
4. Add FastAPI auth, rate limiting, and deployment packaging if API deployment
   is still a goal.
5. Run and document real PPO/RLHF training before advertising PPO support
   beyond reward-model and import smoke diagnostics.
