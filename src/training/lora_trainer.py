"""
src/training/lora_trainer.py

Fine-tunes Qwen2.5-3B using LoRA via PEFT + TRL on Apple Silicon (MPS).
Logs training metrics to both W&B and MLflow.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
import wandb
import yaml
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ALPACA_PROMPT = """Below is an instruction related to aeronautical charts and aviation procedures.
Write a response that accurately answers the question.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


class LoRATrainer:
    """Fine-tunes Qwen2.5-3B with LoRA using PEFT + TRL on Apple Silicon MPS."""

    def __init__(self, config_path: str | Path = "configs/lora_config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None

    # ── Public API ───────────────────────────────────────────────────────────

    def train(
        self,
        train_path: str | Path = "data/synthetic/train.jsonl",
        eval_path: str | Path = "data/synthetic/eval.jsonl",
    ) -> str:
        """
        Run full LoRA fine-tuning pipeline.
        Returns path to saved adapter weights.
        """
        logger.info("=== LoRA Fine-Tuning Pipeline ===")

        # 1. Load base model
        self._load_model()

        # 2. Apply LoRA adapters
        self._apply_lora()

        # 3. Prepare datasets
        train_dataset = self._prepare_dataset(train_path)
        eval_dataset = self._prepare_dataset(eval_path)

        logger.info(
            f"Dataset: {len(train_dataset)} train / {len(eval_dataset)} eval examples"
        )

        # 4. Configure training
        training_args = self._build_training_args()

        # 5. Build trainer
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        # 6. Train (W&B + MLflow optional)
        output_dir = self.config["training"]["output_dir"]
        use_mlflow = bool(os.environ.get("MLFLOW_TRACKING_URI"))
        use_wandb = bool(os.environ.get("WANDB_API_KEY"))

        ctx = mlflow.start_run(run_name="lora-finetune") if use_mlflow else contextlib.nullcontext()
        with ctx:
            if use_mlflow:
                mlflow.log_params(self._flatten_config())
            if use_wandb:
                wandb.init(
                    project=self.config["logging"]["wandb_project"],
                    name="lora-finetune",
                    config=self.config,
                )

            logger.info("Starting LoRA training...")
            trainer_stats = trainer.train()

            final_loss = trainer_stats.training_loss
            if use_mlflow:
                mlflow.log_metric("final_train_loss", final_loss)
            logger.success(f"Training complete. Final loss: {final_loss:.4f}")

            if use_wandb:
                wandb.finish()

        # 7. Save adapter weights
        adapter_path = Path(output_dir) / "adapter"
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        logger.success(f"Adapter saved → {adapter_path}")

        return str(adapter_path)

    def merge_and_export(self, adapter_path: str | Path) -> str:
        """Merge LoRA weights into base model and save full model."""
        if self.model is None:
            self._load_model()
            self._apply_lora()

        merged_path = Path(self.config["training"]["output_dir"]) / "merged"
        logger.info(f"Merging adapter into base model → {merged_path}")

        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_path))
        self.tokenizer.save_pretrained(str(merged_path))
        logger.success(f"Merged model saved → {merged_path}")
        return str(merged_path)

    def export_gguf(self, merged_path: str | Path, quantization: str = "q4_k_m") -> str:
        """Export merged model to GGUF format via llama.cpp convert script."""
        gguf_path = Path("models") / f"aerosense-chartqa-lora-{quantization}.gguf"
        gguf_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting to GGUF ({quantization}) → {gguf_path}")

        # Try llama.cpp conversion (must be installed separately)
        convert_script = Path("llama.cpp/convert_hf_to_gguf.py")
        if not convert_script.exists():
            raise FileNotFoundError(
                "llama.cpp/convert_hf_to_gguf.py not found. "
                "Clone llama.cpp and run `make` first, or install via brew."
            )

        subprocess.run(
            [
                sys.executable, str(convert_script),
                str(merged_path),
                "--outfile", str(gguf_path),
                "--outtype", quantization,
            ],
            check=True,
        )
        logger.success(f"GGUF export complete → {gguf_path}")
        return str(gguf_path)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        model_cfg = self.config["model"]
        logger.info(f"Loading base model: {model_cfg['name']}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            torch_dtype=torch.bfloat16,
        )
        logger.info("Base model loaded ✓")

    def _apply_lora(self) -> None:
        lora_cfg = self.config["lora"]
        logger.info(
            f"Applying LoRA adapters (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})"
        )

        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info("LoRA adapters applied ✓")

    def _prepare_dataset(self, path: str | Path) -> Dataset:
        """Load JSONL and format as Alpaca prompt strings."""
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                text = ALPACA_PROMPT.format(
                    ex["instruction"],
                    ex.get("input", ""),
                    ex["output"],
                ) + self.tokenizer.eos_token
                examples.append({"text": text, **ex})

        return Dataset.from_list(examples)

    def _build_training_args(self) -> SFTConfig:
        t = self.config["training"]
        return SFTConfig(
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            warmup_ratio=t["warmup_ratio"],
            num_train_epochs=t["num_train_epochs"],
            learning_rate=t["learning_rate"],
            fp16=t["fp16"],
            bf16=t["bf16"],
            logging_steps=self.config["logging"]["log_steps"],
            optim="adamw_torch",  # adamw_8bit requires bitsandbytes CUDA; use torch on MPS
            weight_decay=0.01,
            lr_scheduler_type=t["lr_scheduler_type"],
            seed=42,
            output_dir=t["output_dir"],
            report_to=[r for r in ["wandb", "mlflow"] if os.environ.get({"wandb": "WANDB_API_KEY", "mlflow": "MLFLOW_TRACKING_URI"}[r])],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            dataset_text_field="text",
            max_length=t["max_seq_length"],
            dataset_num_proc=2,
            packing=False,
        )

    def _flatten_config(self) -> dict[str, Any]:
        """Flatten nested config for MLflow param logging."""
        flat: dict[str, Any] = {}
        for section, values in self.config.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat[f"{section}.{k}"] = v
        return flat

    @staticmethod
    def _load_config(path: str | Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)


if __name__ == "__main__":
    trainer = LoRATrainer("configs/lora_config.yaml")
    adapter_path = trainer.train()
    merged_path = trainer.merge_and_export(adapter_path)
    gguf_path = trainer.export_gguf(merged_path)
    print(f"\nPipeline complete:\n  Adapter: {adapter_path}\n  Merged: {merged_path}\n  GGUF: {gguf_path}")
