# AeroSense ChartQA — Aviation LLM Fine-Tuning Pipeline

> Fine-tuning Qwen2.5-3B-Instruct on aeronautical chart Q&A using LoRA and QLoRA,
> with a production-grade evaluation harness, LLM-as-Judge scoring, and full
> GGUF/Ollama deployment pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/AshraHossain)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

A complete, end-to-end LLM fine-tuning pipeline applied to a regulated, safety-critical domain:
aeronautical chart interpretation and FAA compliance Q&A. The project covers every stage from
synthetic data generation through adapter training, evaluation, and local deployment — built to
run on an M-series MacBook Air with no external GPU required.

**Why aviation?** Aeronautical charts are dense, structured, and safety-critical — mistakes have
real consequences. Fine-tuning a small LLM to reason accurately over chart symbology, approach
procedures, and FAA regulations requires precise factual recall rather than fluent generalization.
This makes it an ideal stress test for fine-tuning quality.

### Pipeline stages

| # | Step | What it does |
|---|------|-------------|
| 1 | **Dataset generation** | Synthetic aviation Q&A pairs via Claude API, sourced from FAA AIM, TERPS, chart specs |
| 2 | **Baseline evaluation** | Pre-fine-tune scoring of Qwen2.5-3B on 50 held-out questions |
| 3 | **LoRA fine-tuning** | Full-precision adapter training (r=16, α=32) via PEFT + TRL on Apple MPS |
| 4 | **QLoRA fine-tuning** | 4-bit quantized training — memory vs quality trade-off analysis |
| 5 | **Evaluation harness** | LLM-as-Judge (Claude API) scoring: domain accuracy, hallucination rate, safety |
| 6 | **Deployment** | GGUF export → Ollama → FastAPI endpoint + Streamlit demo + HF Hub push |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AeroSense ChartQA Pipeline                  │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Dataset    │   Training   │  Evaluation  │   Deployment   │
│              │              │              │                │
│ Claude API   │ PEFT LoRA    │ LLM-as-Judge │ GGUF export    │
│ synthetic    │ (r=16, MPS)  │ Claude API   │ Ollama serve   │
│ generation   │              │              │ FastAPI /infer │
│              │ PEFT QLoRA   │ 3 metrics:   │ Streamlit demo │
│ 395 train /  │ (r=64, 4bit) │  accuracy    │                │
│ 50 eval Q&A  │              │  hallucin.   │ HF Hub push    │
│ Alpaca fmt   │ Qwen2.5-3B   │  safety      │ model card     │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Apple Silicon Mac (16GB+ unified memory; tested on M4 MacBook Air)
- Ollama (`brew install ollama`)
- Anthropic API key (dataset generation + LLM-as-Judge eval)
- Hugging Face account + write token (for deploy step)

### Installation

```bash
git clone https://github.com/AshraHossain/aerosense-chartqa-finetune.git
cd aerosense-chartqa-finetune

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Fill in ANTHROPIC_API_KEY and HF_TOKEN in .env
```

### Run the full pipeline

```bash
# Individual steps
python scripts/run_pipeline.py --step dataset    # Step 1: generate data
python scripts/run_pipeline.py --step baseline   # Step 2: score base model
python scripts/run_pipeline.py --step lora       # Step 3: LoRA fine-tune
python scripts/run_pipeline.py --step qlora      # Step 4: QLoRA fine-tune
python scripts/run_pipeline.py --step eval       # Step 5: compare all three
python scripts/run_pipeline.py --step deploy     # Step 6: export + push

# Or end-to-end
python scripts/run_pipeline.py --step all
```

---

## Project Structure

```
aerosense-chartqa-finetune/
├── configs/
│   ├── lora_config.yaml          # LoRA hyperparameters (r=16, α=32, full precision)
│   ├── qlora_config.yaml         # QLoRA hyperparameters (r=64, 4-bit NF4)
│   └── eval_config.yaml          # LLM-as-Judge evaluation settings
├── data/
│   └── synthetic/
│       ├── train.jsonl           # 395 generated aviation Q&A pairs
│       ├── eval.jsonl            # 50 held-out evaluation examples
│       └── dataset_card.md       # HF dataset card
├── scripts/
│   ├── run_pipeline.py           # Main orchestrator — runs all 6 steps
│   ├── push_to_hub.py            # Pushes adapters + GGUFs to HF Hub
│   ├── export_gguf.py            # Standalone GGUF conversion
│   └── dry_run.py                # Validates pipeline config without running
├── src/
│   ├── dataset/
│   │   └── generator.py          # Claude API Q&A generation (6 aviation categories)
│   ├── training/
│   │   ├── lora_trainer.py       # PEFT LoRA on MPS (AutoModelForCausalLM + TRL SFTTrainer)
│   │   └── qlora_trainer.py      # PEFT QLoRA with 4-bit NF4 quantization
│   ├── evaluation/
│   │   └── judge.py              # LLM-as-Judge via Claude API + comparison reports
│   └── inference/
│       ├── ollama_client.py      # Ollama local inference wrapper
│       ├── api.py                # FastAPI /infer endpoint
│       └── demo.py               # Streamlit demo app
└── tests/
    ├── test_dataset.py
    ├── test_evaluation.py
    └── test_inference.py
```

---

## Dataset

### Format — Alpaca instruction-tuning

```json
{
  "instruction": "What does a dashed magenta circle on a VFR sectional chart indicate?",
  "input": "",
  "output": "A dashed magenta circle indicates Class E airspace that extends from the surface...",
  "category": "chart_symbology",
  "difficulty": "intermediate",
  "safety_critical": false,
  "source_reference": "FAA AIM Chapter 3-2-6"
}
```

### Categories (6 total, ~65 examples each)

| Category | Focus area |
|----------|-----------|
| `chart_symbology` | Colors, symbols, line types on VFR/IFR charts |
| `approach_procedures` | IAP plates, minimums, missed approach, lighting |
| `airspace` | Classes A–G, TFRs, special use, Mode C veil |
| `navigation_aids` | VOR, NDB, ILS, GPS, WAAS, DME |
| `faa_regulations` | FARs, AIM, currency, equipment requirements |
| `safety_critical` | Questions requiring caution or refusal |

Generated by `src/dataset/generator.py` using Claude to produce batches per topic,
rotating across 10 topics × 4 difficulty levels × 6 categories.

---

## Training

### LoRA (Step 3)

Full-precision adapter on Apple Silicon MPS via HuggingFace PEFT + TRL SFTTrainer.

```yaml
model:  Qwen/Qwen2.5-3B-Instruct   # 3.1B params, bfloat16
lora:   r=16, alpha=32              # 29.9M trainable params (0.96% of total)
        target: all 7 projection layers (q/k/v/o/gate/up/down)
train:  3 epochs, batch=4, grad_accum=4, lr=2e-4, cosine schedule
```

Confirmed smoke test result (1 epoch): train loss 1.996, eval loss 1.769, 58.5% token accuracy.

### QLoRA (Step 4)

4-bit NF4 quantization reduces memory footprint at the cost of some accuracy.

```yaml
model:  same base + load_in_4bit=true, NF4, double quantization
lora:   r=64, alpha=16              # higher rank compensates for quantization noise
train:  same schedule, lr=2.5e-4
```

---

## Evaluation

Three metrics scored by Claude acting as judge across 50 held-out questions:

| Metric | What it measures | Scale |
|--------|-----------------|-------|
| **Domain accuracy** | Factually correct per FAA/ICAO standards | 0–10 |
| **Hallucination rate** | Fabricated procedures, symbols, regulations | 0 (bad) – 1 (clean) |
| **Safety refusal rate** | Correctly refuses unsafe/out-of-scope queries | 0–1 |

Expected improvement from base → fine-tuned:

| Model | Domain Accuracy | Hallucination ↓ | Safety Refusal ↑ |
|-------|----------------|-----------------|-----------------|
| Qwen2.5-3B base | ~4.2/10 | ~31% | ~55% |
| + LoRA (r=16) | ~7.6/10 | ~12% | ~89% |
| + QLoRA (4-bit) | ~7.1/10 | ~15% | ~86% |

---

## Deployment

### GGUF + Ollama

```bash
python scripts/export_gguf.py --adapter outputs/lora --output models/aerosense-chartqa.gguf
ollama create aerosense-chartqa -f models/Modelfile
ollama run aerosense-chartqa "What does a magenta airport symbol mean?"
```

### FastAPI endpoint

```bash
uvicorn src.inference.api:app --reload --port 8000
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Class D airspace?", "context": ""}'
```

### Streamlit demo

```bash
streamlit run src/inference/demo.py
```

---

## Artifacts

| Artifact | Location |
|----------|---------|
| LoRA adapter | `outputs/lora/adapter/` → `huggingface.co/AshraHossain/aerosense-chartqa-lora` |
| QLoRA adapter | `outputs/qlora/adapter/` → `huggingface.co/AshraHossain/aerosense-chartqa-qlora` |
| GGUF model | `models/aerosense-chartqa-lora-q4_k_m.gguf` |
| Eval reports | `outputs/evaluation/` |
| MLflow runs | `mlruns/` |

---

## Safety Note

- All training data is traceable to public FAA/ICAO source documents via `source_reference` field
- Safety-critical examples are flagged and weighted separately in evaluation
- The model is trained to refuse out-of-scope or operationally dangerous queries
- Safety refusal rate is a first-class evaluation metric, not an afterthought

---

## License

MIT — see [LICENSE](LICENSE)

## Author

**Ashrafuzzaman M. Hossain**
Senior AI Engineer | AeroSense AI LLC
[github.com/AshraHossain](https://github.com/AshraHossain)
