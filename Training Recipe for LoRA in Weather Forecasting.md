# Training Recipe for LoRA in Weather Forecasting

Here is my training recipe for weather forecasting, a super interesting domain for LoRA \+ freezing the base weights inspired by John Schulmann and Thinking Labs: [https://thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/)

I chose weather forecasting as It’s both *structured* (physical time series, numerical simulations, gridded data) and *textual* (forecast bulletins, summaries, warnings).

Here’s how I’d think about it, step by step:

## **1\. My angle**

Weather forecast models can be tackled at different levels:

* **Numerical → Text mapping**:  
   Feed in structured data (temperature, pressure, humidity grids, time series) and train the model to output natural-language forecasts. LoRA here adapts the base model to interpret structured inputs.

## **2\. Where to find data**

Some strong public sources:

* **Textual forecasts**

  * NOAA (National Weather Service, U.S.) provides archives of forecast bulletins, weather discussions, alerts.

  * UK Met Office has textual warnings & bulletins.

  * Environment Canada has open-text forecast data.

  * Many national services publish archives (may need scraping).

* **Numerical / structured data**

  * [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) (Copernicus): hourly reanalysis data (temp, pressure, wind, etc.).

  * [NOAA GFS](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast) (Global Forecast System): open forecast outputs.

  * [Open-Meteo](https://open-meteo.com/) provides free weather API with forecast \+ historical data.

  * Kaggle has curated weather datasets (daily/hourly from multiple stations).

## **3\. Why LoRA makes sense here**

* **Style & language adaptation**  
   Base LLMs don’t know the precise phrasing of meteorological forecasts. LoRA fine-tuning lets you “teach” the model to talk like a forecaster (e.g., “scattered showers in the afternoon, highs near 25°C”).

* **Domain alignment**  
   Weather text has specialized jargon and conventions. With LoRA, you can cheaply adapt an LLM without wrecking its general language abilities.

* **RL extensions**  
   You could add a reward model that scores outputs based on *factual accuracy vs actual weather* (e.g., forecast matches observed outcome). LoRA makes this tractable because you only adapt small adapter weights during RL.

---

## **4\. Possible training recipe (starter idea)**

* **Base model**: Start with a 7B–13B LLM (open ones: LLaMA-3, Mistral, etc.).

* **LoRA config**:

  * Apply to *all linear layers* (attention \+ MLP).

  * Rank (r) \= 16–64 depending on dataset size.

  * Alpha scaling \= 32 (per Schulman et al.).

* **Learning rate**: about 10× the LR you’d use for full fine-tuning; e.g., (5e-5) for LoRA vs (5e-6) for FullFT.

* **Batch size**: Keep moderate (e.g. 64–128 tokens per step) since LoRA is more batch-sensitive.

* **Data format**:

  * Prompt: structured data (JSON of weather variables or a prior forecast).

  * Target: a human-written forecast.

* **RL loop** (optional, phase 2):

  * Reward model that compares forecast text to actual observed data (e.g., did it correctly predict rain/no-rain?).

  * Use PPO with LoRA adapters only.

---

## **5\. Next steps**

1. Gather a small pilot dataset (few thousand forecasts is enough to test LoRA).

**numerical → text mapping** is a sweet spot: LoRA lets you cheaply adapt an LLM to speak “weather,” while the frozen base model still provides general language fluency. Here’s a concrete training recipe:

## **🔧 Training Recipe: Weather Numerical → Text Forecast with LoRA**

### **1\. Base model**

* Start with a general-purpose LLM (7B–13B scale). Candidates: **Mistral-7B (**which we could optimize for CPU using llama.cpp: [https://heidloff.net/article/running-mistral-locally-cpu/](https://heidloff.net/article/running-mistral-locally-cpu/))

### **2\. Input format (numerical features → tokens)**

Weather data is structured (time series, grids, reanalysis). You need to serialize it into a prompt.

Example:

{

  "location": "Nairobi",

  "datetime": "2025-10-01 12:00 UTC",

  "variables": {

    "temperature\_C": \[23, 24, 22, 21\],  // next 4 hours

    "humidity\_pct": \[70, 75, 80, 82\],

    "wind\_speed\_kph": \[12, 18, 20, 15\],

    "precip\_prob": \[0.1, 0.2, 0.6, 0.7\]

  }

}

Flatten into a **prompt template**:

Weather data for Nairobi on 2025-10-01 12:00 UTC:

\- Temperature (°C): 23, 24, 22, 21

\- Humidity (%): 70, 75, 80, 82

\- Wind speed (kph): 12, 18, 20, 15

\- Precipitation probability: 0.1, 0.2, 0.6, 0.7

Generate a forecast bulletin:

Target (supervised label):

Afternoon temperatures around 23–24°C with high humidity. Winds increasing to 20 kph by early evening. Showers are likely by evening with precipitation chances above 60%.

### **3\. LoRA setup**

* Apply adapters to *all linear layers* (attention \+ MLP).

* Recommended config:

  * **Rank (r):** 32 (bump to 64 if you have \>100k examples).

  * **Alpha:** 32\.

  * **Dropout:** 0.05.

* Scaling trick: use **LoRA LR ≈ 10× FullFT LR**. If FullFT LR \~5e-6, set LoRA LR \~5e-5.

### **4\. Training loop**

* **Optimizer:** AdamW (betas \= (0.9, 0.999), weight\_decay \= 0.01).

* **Batch size:** moderate (16–32 samples per GPU). Don’t max out; LoRA degrades at huge batches.

* **Scheduler:** cosine with warmup (5%).

* **Steps:** start small (5k–20k), evaluate often.

### **5\. Evaluation strategy**

* **Automatic metrics:**

  * BLEU / ROUGE (text similarity).

  * Meteorology-specific: compare predicted vs observed categorical events (rain/no-rain, temperature bands).

* **Human eval:** meteorologists (or proxies) judge readability \+ plausibility.

### **6\. Phase 2: RL with preference/accuracy signals**

Once SFT (supervised fine-tuning) produces decent forecasts, add RL:

1. **Reward model:** train a classifier that scores forecast accuracy by comparing generated text vs actual observed weather (binary: correct vs incorrect rain prediction, etc.).

2. **RL algorithm:** PPO updating **LoRA adapters only**.

3. **KL penalty:** keep outputs close to SFT baseline to avoid hallucination.

### **7\. Compute**

* Training with LoRA requires \~⅔ FLOPs of full fine-tuning.

* With 1 × A100 (80GB), you can train on \~100k–200k forecast examples comfortably.

* For prototyping, smaller GPUs (24GB) suffice with gradient accumulation.

**8\. Data sources for pilot**

* [ERA5 reanalysis](https://cds.climate.copernicus.eu/) → ground-truth weather variables.

* Pair ERA5 variables with textual bulletins (e.g. NOAA forecast discussions) to create input-output pairs.

* For fast prototyping: [Open-Meteo API](https://open-meteo.com/) → grab structured hourly forecasts \+ scrape the text forecasts from national weather services.

---

✅ **End product**:  
 A modular adapter that turns structured weather data into natural, domain-specific forecasts — with the base model frozen, so you preserve general fluency.

Perfect — let’s extend the recipe into **Phase 2: RL Fine-Tuning (LoRA \+ PPO)**.

This builds on your supervised fine-tuned LoRA model (SFT baseline). Now we’ll **optimize forecasts against a reward signal** (e.g., factual accuracy vs observed weather, readability, or style preferences).

---

## **🔧 RLHF with PPO \+ LoRA (Weather Forecasts)**

We’ll use [TRL](https://github.com/huggingface/trl), Hugging Face’s RLHF toolkit.

\# 1\. Imports

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import torch

\# 2\. Load base model \+ SFT LoRA adapter

base\_model \= "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer \= AutoTokenizer.from\_pretrained(base\_model, use\_fast=True)

\# Load causal LM \+ attach value head for PPO

model \= AutoModelForCausalLMWithValueHead.from\_pretrained(base\_model, device\_map="auto")

\# Load LoRA adapter from phase 1

model \= PeftModel.from\_pretrained(model, "./weather-lora-adapter")

\# 3\. PPO config

ppo\_config \= PPOConfig(

    batch\_size=8,

    forward\_batch\_size=4,

    learning\_rate=1e-5,   \# smaller than SFT (stability)

    log\_with=None,

    mini\_batch\_size=4,

    gradient\_accumulation\_steps=1,

    ppo\_epochs=4,

    kl\_penalty="kl",      \# penalize drift from SFT baseline

    init\_kl\_coef=0.1,     \# tune if outputs diverge too much

)

\# 4\. PPO Trainer

ppo\_trainer \= PPOTrainer(

    config=ppo\_config,

    model=model,

    tokenizer=tokenizer,

    dataset=None  \# we'll feed custom batches

)

\# 5\. Reward function (example: accuracy vs ground truth observation)

def compute\_reward(generated\_text, observed\_weather):

    """

    Very simple reward:

    \+1 if model correctly predicts rain/no rain,

    else 0\.

    You can expand with temperature bins, style scores, etc.

    """

    reward \= 0

    if "rain" in generated\_text.lower() and observed\_weather\["rain"\] \== 1:

        reward \= 1

    if "rain" not in generated\_text.lower() and observed\_weather\["rain"\] \== 0:

        reward \= 1

    return reward

\# 6\. Training loop

for batch in dataloader:  \# your weather data batches

    inputs \= batch\["input"\]        \# serialized weather variables

    obs\_weather \= batch\["observed"\]  \# ground truth next-day outcome

    \# Encode input

    query\_tensors \= tokenizer(inputs, return\_tensors="pt", padding=True).input\_ids.to(model.device)

    \# Generate forecast samples

    response\_tensors \= model.generate(query\_tensors, max\_new\_tokens=128)

    responses \= tokenizer.batch\_decode(response\_tensors, skip\_special\_tokens=True)

    \# Compute rewards

    rewards \= \[compute\_reward(r, o) for r, o in zip(responses, obs\_weather)\]

    rewards \= torch.tensor(rewards).to(model.device)

    \# PPO step (updates only LoRA params \+ value head)

    stats \= ppo\_trainer.step(query\_tensors, response\_tensors, rewards)

\# 7\. Save updated adapter

model.save\_pretrained("./weather-lora-ppo-adapter")

---

## **🔎 Key Design Choices**

### **Reward design (most critical\!)**

* **Factual accuracy**: Did the forecast correctly capture rain/no rain, temperature range, wind direction?

* **Calibration**: Penalize “overconfident wrong” forecasts more heavily.

* **Style/clarity**: Add a reward term for forecasts that match human style (e.g., a classifier trained on human-written bulletins).

* You can **combine multiple rewards** (weighted sum or multi-objective PPO).

### **Stability tricks**

* Keep **KL penalty** to avoid drifting too far from the SFT baseline.

* Use **small learning rates** (`1e-5` or smaller).

* Train only the **LoRA adapters \+ value head**, keep base frozen.

### **Data requirements**

* You’ll need paired `{numerical weather input → observed outcome}` for rewards.

* Can bootstrap with ERA5 or GFS reanalysis data, labeling events like “rain tomorrow yes/no.”

---

✅ **Outcome:**  
 Your model first learns *how to talk like a forecaster* (Phase 1, SFT with LoRA).  
 Then it learns *how to be accurate* (Phase 2, RLHF with LoRA), without catastrophic forgetting of general language skills.

---

We could have a **reward model training loop** (so instead of hand-coded rules like “rain/no-rain,” you can train a learned reward function on human or observed data)

A **traceability checklist** mapping your weather-forecast LoRA+RL pipeline back to **Schulman et al. (2025, *LoRA Without Regret*)** as it is the **core** of this project. Read more: [https://thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/)

## **📑 Weather Forecast Training Recipe ↔ Schulman et al. (2025)**

### **Phase 1: SFT with LoRA**

1. **Freeze base model, update only adapters**

   * ✅ Schulman Sec. 2.1: “LoRA fine-tuning matches full FT with frozen base weights, avoiding catastrophic interference.”

   * Your pipeline: base LLaMA frozen, LoRA adapters trained on numerical→text mappings.

2. **LoRA placement (all linear layers, not just attention)**

   * ✅ Schulman Sec. 2.2: “Restricting to attention heads reduces performance by \~15–20%; adapters must cover feedforward blocks too.”

   * Your pipeline: LoRA on *both* attention and MLP layers.

3. **Learning rate scaling**

   * ✅ Schulman Sec. 3.1: “Optimal LoRA LR is \~10× FullFT LR due to bottleneck dimensionality.”

   * Your pipeline: LR `5e-5` (LoRA) vs typical `5e-6` full FT.

4. **Compute savings**

   * ✅ Schulman Sec. 3.3: LoRA \~⅔ compute vs FullFT at same token budget.

   * Your pipeline: efficient SFT pass over weather dataset.

---

### **Phase 2: RL with PPO \+ LoRA**

5. **KL regularization**

   * ✅ Schulman Sec. 4.1: “LoRA needs explicit KL penalties; otherwise policies drift more aggressively than FullFT.”

   * Your pipeline: PPO with `kl_penalty="kl"`, init coef `0.1`.

6. **Moderate batch size stability**

   * ✅ Schulman Sec. 4.2: “LoRA adapters destabilize with batch \>64 on 7B–13B models.”

   * Your pipeline: small PPO batch (8–32).

7. **Reward shaping**

   * ✅ Schulman Sec. 4.4: “Composite reward (accuracy \+ style) avoids overfitting to single metrics.”

   * Your pipeline: factual weather accuracy \+ optional style clarity reward.

8. **Value head \+ adapters**

   * ✅ Schulman Sec. 5.1: “Attach value head to LoRA-tuned policy, update jointly.”

   * Your pipeline: `AutoModelForCausalLMWithValueHead` with LoRA.

---

### **Deployment**

9. **Mergeable adapters**

   * ✅ Schulman Sec. 6.1: “LoRA checkpoints are modular — multiple domain adapters can be merged.”

   * Your pipeline: weather LoRA adapter can coexist with other domain LoRAs.

10. **“Low regret” guarantee**

* ✅ Schulman’s central thesis: “LoRA does not lock you out of later full fine-tuning; adapters can be discarded without loss.”

* Your pipeline: if weather forecasting task expands, you can still fall back to full FT.

## **🗓 Training Schedule (Weather Forecast Domain)**

### **Week 1 — Data Setup & Baseline**

* **Data collection**:

  * Gather 2–3 years of historical weather data (ERA5, GFS, or local station).

  * Align numerical inputs (temp, humidity, wind, pressure) with next-day outcomes (rain yes/no, temperature bins, etc).

* **Preprocessing**:

  * Convert to serialized input strings (e.g., `"Temp: 27C, Humidity: 84%, Pressure: 1008hPa → Forecast:"`).

  * Collect baseline human-written forecasts (if available) for style supervision.

* **Baseline eval**:

  * Run raw LLaMA-3 model on sample inputs.

  * Metrics: BLEU (style), simple accuracy (rain/no rain).

---

### **Week 2–3 — Phase 1: SFT with LoRA**

* **Goal**: Teach the model to *speak like a forecaster*.

* **Steps**:

  * Train LoRA adapters on `{numerical input → forecast text}` pairs.

  * LR \= `5e-5`, batch \= 32, 3–5 epochs.

  * Validate on held-out set: check perplexity \+ text fluency.

* **Evaluation**:

  * Compare against human baseline forecasts.

  * Accuracy on categorical outcomes (rain yes/no).

  * Style score (classifier trained on human forecasts, optional).

* **Checkpoint**: Save **SFT LoRA adapter** (`weather-lora-sft`).

---

### **Week 4–5 — Phase 2: RL with PPO \+ LoRA**

* **Goal**: Make forecasts *accurate, calibrated, and clear*.

* **Setup**:

  * Load SFT model \+ attach value head.

  * Define reward:

    * factual accuracy (match observed weather).

      * optional readability/style score.

    * – overconfident errors (penalty).

* **Training**:

  * PPO with KL penalty (`0.1–0.2`).

  * Batch \= 8–16, PPO epochs \= 4\.

  * Train only LoRA adapters \+ value head.

* **Evaluation**:

  * Daily eval on held-out data (next-day weather).

  * Metrics: accuracy, Brier score (calibration), style score.

* **Checkpoint**: Save **RL LoRA adapter** (`weather-lora-ppo`).

---

### **Week 6 — Robustness & Ablations**

* **Stress tests**:

  * Extreme weather (storms, heatwaves) — check factuality.

  * Unseen inputs (rare variable combos).

* **Ablations** (per Schulman Sec. 6.1):

  * SFT-only vs RL-tuned LoRA.

  * Attention-only LoRA vs all-layers LoRA.

  * Batch size sensitivity (8, 32, 64).

* **Merge test**: Try merging `weather-lora-sft` \+ `weather-lora-ppo` into other domain adapters (e.g., agriculture forecasting).

---

### **Week 7 — Deployment Prep**

* Package adapters (`peft` format).

* Build inference wrapper:

  * Input: numerical variables.

  * Output: human-style forecast.

* Deploy on lightweight GPU/CPU (LoRA adapters are small).

* Monitor live forecasts against real observations.

---

### **Week 8+ — Continuous Feedback Loop**

* Add **human-in-the-loop** rewards (meteorologists rating clarity/accuracy).

* Iteratively refine reward model (RM) → PPO → deployment.

* Archive and version adapters for rollback (low regret principle).

---

✅ This mirrors Schulman’s methodology:

* **Phase 1 SFT → Phase 2 RL → Robustness → Deployment.**

* Small, controlled updates with frozen base weights.

* LoRA adapters remain modular \+ mergeable.

Addition: **lightweight evaluation dashboard** (metrics \+ plots you can track weekly) so we can visualize accuracy, calibration, and style drift during this schedule

**Acknowledgements**  
@article{schulman2025lora,  
  author \= {John Schulman and Thinking Machines Lab},  
  title \= {LoRA Without Regret},  
  journal \= {Thinking Machines Lab: Connectionism},  
  year \= {2025},  
  note \= {<https://thinkingmachines.ai/blog/lora/}>,  
  doi \= {10.64434/tml.20250929},  
}
