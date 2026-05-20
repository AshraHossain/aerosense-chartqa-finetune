# AeroSense ChartQA — Aviation LLM Fine-Tuning Pipeline

> Fine-tuning Qwen2.5-3B on aeronautical chart Q&A using LoRA and QLoRA with a production-grade
> evaluation harness, LLM-as-Judge scoring, and full GGUF/Ollama deployment pipeline.

[![CI](https://github.com/AshraHossain/aerosense-chartqa-finetune/actions/workflows/ci.yml/badge.svg)](https://github.com/AshraHossain/aerosense-chartqa-finetune/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/AshraHossain)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project demonstrates a full LLM fine-tuning lifecycle applied to a regulated,
safety-critical aviation domain — aeronautical chart interpretation and FAA compliance Q&A.

**Why this domain?** Aeronautical charts are dense, structured, and safety-critical — mistakes
have real-world consequences. Fine-tuning a small LLM to reason accurately over chart symbology,
approach procedures, and FAA regulations requires the model to learn precise, factual recall
rather than fluent generalization. This makes it an ideal stress test for fine-tuning quality.

### What this project covers

| Phase | Description |
|-------|-------------|
| **Dataset generation** | Synthetic aviation Q&A pairs via Claude API from FAA AIM, TERPS, chart specs |
| **Baseline evaluation** | Pre-fine-tune scoring of Qwen2.5-3B on held-out eval set |
| **LoRA fine-tuning** | Full-precision adapter training (r=16, alpha=32) via Unsloth |
| **QLoRA fine-tuning** | 4-bit quantized training via bitsandbytes — memory/quality trade-off analysis |
| **Evaluation harness** | LLM-as-Judge (Claude API) scoring: accuracy, hallucination rate, safety refusal rate |
| **Deployment pipeline** | GGUF export → Ollama local inference → FastAPI endpoint → Streamlit demo |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AeroSense ChartQA Pipeline                  │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Dataset    │   Training   │  Evaluation  │   Deployment   │
│              │              │              │                │
│ Claude API   │ Unsloth      │ LLM-as-Judge │ GGUF export    │
│ synthetic    │ LoRA (r=16)  │ Claude API   │ Ollama serve   │
│ generation   │ QLoRA (4bit) │ MLflow logs  │ FastAPI /infer │
│              │ W&B tracking │ W&B metrics  │ Streamlit demo │
│ ~500–1K      │              │              │                │
│ Q&A pairs    │ Qwen2.5-3B   │ 50 held-out  │ model card     │
│ Alpaca fmt   │ base model   │ questions    │ HF Hub push    │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- 24GB+ unified memory (M4 MacBook Air or equivalent)
- Ollama installed (`brew install ollama`)
- Anthropic API key (for dataset generation + LLM-as-Judge eval)
- Weights & Biases account (free tier works)
- Hugging Face account + write token

### Installation

```bash
git clone https://github.com/AshraHossain/aerosense-chartqa-finetune.git
cd aerosense-chartqa-finetune

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run the full pipeline

```bash
# Step 1: Generate synthetic dataset
python scripts/run_pipeline.py --step dataset

# Step 2: Baseline evaluation (pre fine-tune)
python scripts/run_pipeline.py --step baseline

# Step 3: LoRA fine-tuning
python scripts/run_pipeline.py --step lora

# Step 4: QLoRA fine-tuning
python scripts/run_pipeline.py --step qlora

# Step 5: Evaluation harness (compare all three)
python scripts/run_pipeline.py --step eval

# Step 6: Export and deploy
python scripts/run_pipeline.py --step deploy

# Or run everything end to end
python scripts/run_pipeline.py --step all
```

---

## Project Structure

```
aerosense-chartqa-finetune/
├── data/
│   ├── raw/                    # Source FAA/TERPS reference docs (PDFs, text)
│   ├── processed/              # Cleaned, chunked source material
│   └── synthetic/              # Generated Q&A pairs (JSONL)
│       ├── train.jsonl         # ~450 training examples
│       ├── eval.jsonl          # ~50 held-out evaluation examples
│       └── dataset_card.md     # Hugging Face dataset card
├── src/
│   ├── dataset/
│   │   ├── generator.py        # Claude API synthetic Q&A generation
│   │   ├── validator.py        # Dataset quality checks + dedup
│   │   └── formatter.py        # Alpaca format conversion
│   ├── training/
│   │   ├── lora_trainer.py     # LoRA fine-tuning with Unsloth
│   │   ├── qlora_trainer.py    # QLoRA (4-bit) fine-tuning
│   │   └── callbacks.py        # W&B + MLflow logging callbacks
│   ├── evaluation/
│   │   ├── judge.py            # LLM-as-Judge via Claude API
│   │   ├── metrics.py          # Accuracy, hallucination, safety scoring
│   │   └── reporter.py         # Comparison report generator
│   └── inference/
│       ├── api.py              # FastAPI inference endpoint
│       ├── ollama_client.py    # Ollama local inference wrapper
│       └── demo.py             # Streamlit demo app
├── configs/
│   ├── lora_config.yaml        # LoRA hyperparameters
│   ├── qlora_config.yaml       # QLoRA hyperparameters
│   └── eval_config.yaml        # Evaluation settings
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_results_comparison.ipynb
├── scripts/
│   ├── run_pipeline.py         # Main pipeline orchestrator
│   ├── export_gguf.py          # GGUF conversion + Ollama push
│   └── push_to_hub.py          # HF Hub model card + upload
├── tests/
│   ├── test_dataset.py
│   ├── test_evaluation.py
│   └── test_inference.py
├── .github/workflows/
│   └── ci.yml                  # Lint, type-check, unit tests
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── Makefile
└── README.md
```

---

## Dataset

### Format (Alpaca instruction-tuning)

```json
{
  "instruction": "What does a dashed magenta circle on a VFR sectional chart indicate?",
  "input": "",
  "output": "A dashed magenta circle on a VFR sectional chart indicates Class E airspace that extends from the surface. This designation means Class E surface area airspace exists around that airport, typically associated with instrument approach procedures. Pilots must have ATC clearance to operate in this airspace under IFR conditions, and VFR pilots should be aware of increased IFR traffic in the vicinity.",
  "category": "chart_symbology",
  "source": "FAA AIM Chapter 3",
  "difficulty": "intermediate",
  "safety_critical": true
}
```

### Categories

| Category | Description | Examples |
|----------|-------------|---------|
| `chart_symbology` | Chart symbols, colors, line types | "What does a blue airport symbol indicate?" |
| `approach_procedures` | IAP plates, minimums, missed approach | "What is a MALSR?" |
| `airspace` | Classes A–G, special use, TFRs | "What are Class B requirements?" |
| `navigation_aids` | VORs, NDBs, ILS, GPS | "What is a DME arc procedure?" |
| `faa_regulations` | FARs, AIM references, compliance | "What weather minimums apply for VFR?" |
| `safety_critical` | Questions requiring refusal or caution | Adversarial / out-of-scope inputs |

### Generation pipeline

```python
# Claude API generates Q&A from aviation source material
generator = DatasetGenerator(
    model="claude-sonnet-4-20250514",
    source_docs=["FAA_AIM.txt", "TERPS_criteria.txt"],
    n_examples=500,
    categories=AVIATION_CATEGORIES,
    safety_check=True        # Flags safety-critical examples
)
dataset = generator.generate()
```

---

## Training

### LoRA Configuration

```yaml
# configs/lora_config.yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  load_in_4bit: false

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  fp16: false
  bf16: true              # M4 supports bfloat16
  max_seq_length: 2048
  output_dir: "./outputs/lora"

logging:
  wandb_project: "aerosense-chartqa"
  mlflow_experiment: "lora-finetune"
  log_steps: 10
```

### QLoRA Configuration

```yaml
# configs/qlora_config.yaml  (same as lora except:)
model:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

lora:
  r: 64                   # Higher rank compensates for quantization
  lora_alpha: 16

training:
  output_dir: "./outputs/qlora"
```

---

## Evaluation

### LLM-as-Judge scoring

Each model response is scored by Claude on three axes:

| Metric | Description | Scoring |
|--------|-------------|---------|
| **Domain accuracy** | Factually correct per FAA/ICAO standards | 0–10 |
| **Hallucination rate** | Fabricated chart symbols, procedures, regs | 0 = hallucinated, 1 = grounded |
| **Safety refusal rate** | Correctly refuses unsafe/out-of-scope queries | 0–1 |

### Sample results (target after fine-tuning)

| Model | Domain Accuracy | Hallucination Rate | Safety Refusal |
|-------|----------------|-------------------|----------------|
| Qwen2.5-3B base | ~42% | ~31% | ~55% |
| + LoRA (r=16) | ~76% | ~12% | ~89% |
| + QLoRA (4-bit) | ~71% | ~15% | ~86% |

---

## Deployment

### Local Ollama inference

```bash
# After GGUF export
python scripts/export_gguf.py --adapter outputs/lora --output models/aerosense-chartqa.gguf
ollama create aerosense-chartqa -f models/Modelfile
ollama run aerosense-chartqa "What does a magenta airport symbol mean?"
```

### FastAPI endpoint

```bash
uvicorn src.inference.api:app --reload --port 8000

# POST /infer
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Class D airspace?", "context": ""}'
```

### Streamlit demo

```bash
streamlit run src/inference/demo.py
```

---

## Results & Artifacts

| Artifact | Location |
|----------|----------|
| Dataset | `huggingface.co/datasets/AshraHossain/aerosense-chartqa` |
| LoRA adapter weights | `huggingface.co/AshraHossain/qwen2.5-3b-aerosense-lora` |
| QLoRA adapter weights | `huggingface.co/AshraHossain/qwen2.5-3b-aerosense-qlora` |
| GGUF model | `huggingface.co/AshraHossain/qwen2.5-3b-aerosense-gguf` |
| W&B training runs | `wandb.ai/ashrahossain/aerosense-chartqa` |
| MLflow experiments | `./mlruns/` |

---

## Safety & Compliance Note

This project applies DO-178C-inspired engineering discipline to LLM fine-tuning:
- All training data is traceable to public FAA/ICAO source documents
- Safety-critical examples are flagged and weighted in evaluation
- The model includes explicit refusal training for out-of-scope or dangerous queries
- Evaluation harness measures safety refusal rate as a first-class metric — not an afterthought

This approach directly mirrors regulated software verification practices applied to AI systems.

---

## License

MIT — See [LICENSE](LICENSE)

## Author

**Ashrafuzzaman M. Hossain**  
Senior AI Engineer | AeroSense AI LLC  
[linkedin.com/in/ashrafmhossain](https://linkedin.com/in/ashrafmhossain) · [github.com/AshraHossain](https://github.com/AshraHossain)
