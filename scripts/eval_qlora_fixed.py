"""
scripts/eval_qlora_fixed.py

One-off: evaluate the alpha/r-fixed QLoRA retrain against the held-out eval
set, for direct before/after comparison against the original qlora run.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger  # noqa: E402

from src.evaluation.judge import LLMJudge  # noqa: E402
from src.inference.ollama_client import OllamaClient  # noqa: E402

client = OllamaClient(model="aerosense-chartqa-qlora-fixed")
judge = LLMJudge(output_dir="outputs/evaluation")

summary = judge.evaluate_model(
    model_name="qlora_fixed",
    eval_path="data/synthetic/eval.jsonl",
    inference_fn=client.generate,
)
logger.success(
    f"qlora_fixed — accuracy={summary.mean_domain_accuracy:.2f}/10 "
    f"hallucination={summary.mean_hallucination_score:.2f} "
    f"safety={summary.mean_safety_refusal_score:.2f}"
)
