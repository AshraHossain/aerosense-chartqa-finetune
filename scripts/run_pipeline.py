"""
scripts/run_pipeline.py

Main pipeline orchestrator for AeroSense ChartQA fine-tuning.

Usage:
  python scripts/run_pipeline.py --step dataset
  python scripts/run_pipeline.py --step baseline
  python scripts/run_pipeline.py --step lora
  python scripts/run_pipeline.py --step qlora
  python scripts/run_pipeline.py --step eval
  python scripts/run_pipeline.py --step deploy
  python scripts/run_pipeline.py --step all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Step functions ────────────────────────────────────────────────────────────

def step_dataset() -> None:
    logger.info("── Step 1: Dataset Generation ──")
    from src.dataset.generator import DatasetGenerator

    gen = DatasetGenerator(n_train=450, n_eval=50)
    dataset = gen.generate()
    logger.success(
        f"Dataset ready: {len(dataset['train'])} train / {len(dataset['eval'])} eval"
    )


def step_baseline() -> None:
    logger.info("── Step 2: Baseline Evaluation (pre fine-tune) ──")
    from src.evaluation.judge import LLMJudge
    from src.inference.ollama_client import OllamaClient

    # Pull base model via Ollama
    import subprocess
    subprocess.run(["ollama", "pull", "qwen2.5:3b"], check=True)

    base_client = OllamaClient(model="qwen2.5:3b")
    judge = LLMJudge(output_dir="outputs/evaluation")

    summary = judge.evaluate_model(
        model_name="base",
        eval_path="data/synthetic/eval.jsonl",
        inference_fn=base_client.generate,
    )
    logger.success(f"Baseline complete: accuracy={summary.mean_domain_accuracy:.2f}/10")


def step_lora() -> None:
    logger.info("── Step 3: LoRA Fine-Tuning ──")
    from src.training.lora_trainer import LoRATrainer

    trainer = LoRATrainer("configs/lora_config.yaml")
    adapter_path = trainer.train()
    merged_path = trainer.merge_and_export(adapter_path)
    gguf_path = trainer.export_gguf(merged_path)

    logger.success(
        f"LoRA complete:\n"
        f"  Adapter → {adapter_path}\n"
        f"  Merged  → {merged_path}\n"
        f"  GGUF    → {gguf_path}"
    )


def step_qlora() -> None:
    logger.info("── Step 4: QLoRA Fine-Tuning (4-bit) ──")
    from src.training.qlora_trainer import QLoRATrainer

    trainer = QLoRATrainer("configs/qlora_config.yaml")
    adapter_path = trainer.train()
    merged_path = trainer.merge_and_export(adapter_path)
    gguf_path = trainer.export_gguf(merged_path)

    logger.success(
        f"QLoRA complete:\n"
        f"  Adapter → {adapter_path}\n"
        f"  Merged  → {merged_path}\n"
        f"  GGUF    → {gguf_path}"
    )


def step_eval() -> None:
    logger.info("── Step 5: Full Evaluation Comparison ──")
    from src.evaluation.judge import LLMJudge
    from src.inference.ollama_client import OllamaClient

    judge = LLMJudge(output_dir="outputs/evaluation")

    # Evaluate all three models
    summaries = []
    for model_name, ollama_model in [
        ("base",  "qwen2.5:3b"),
        ("lora",  "aerosense-chartqa-lora"),
        ("qlora", "aerosense-chartqa-qlora"),
    ]:
        try:
            client = OllamaClient(model=ollama_model)
            summary = judge.evaluate_model(
                model_name=model_name,
                eval_path="data/synthetic/eval.jsonl",
                inference_fn=client.generate,
            )
            summaries.append(summary)
        except Exception as e:
            logger.warning(f"Skipping {model_name}: {e}")

    if summaries:
        df = judge.compare_models(summaries)
        logger.success(f"\nComparison:\n{df.to_string()}")


def step_deploy() -> None:
    logger.info("── Step 6: Export & Deploy ──")

    # Push to Hugging Face Hub
    from scripts.push_to_hub import push_all
    push_all()

    logger.success(
        "Deployment complete:\n"
        "  → Start API: uvicorn src.inference.api:app --reload --port 8000\n"
        "  → Start demo: streamlit run src/inference/demo.py"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

STEPS = {
    "dataset":  step_dataset,
    "baseline": step_baseline,
    "lora":     step_lora,
    "qlora":    step_qlora,
    "eval":     step_eval,
    "deploy":   step_deploy,
}

def main() -> None:
    parser = argparse.ArgumentParser(description="AeroSense ChartQA Pipeline")
    parser.add_argument(
        "--step",
        choices=[*STEPS.keys(), "all"],
        required=True,
        help="Pipeline step to run",
    )
    args = parser.parse_args()

    if args.step == "all":
        for name, fn in STEPS.items():
            logger.info(f"\n{'='*50}\nRunning: {name}\n{'='*50}")
            fn()
    else:
        STEPS[args.step]()


if __name__ == "__main__":
    main()
