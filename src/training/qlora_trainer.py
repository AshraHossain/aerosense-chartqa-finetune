"""
src/training/qlora_trainer.py

Fine-tunes Qwen2.5-3B with QLoRA configuration via PEFT + TRL on Apple Silicon MPS.
NF4 4-bit quantization (bitsandbytes) is not available on MPS; loads in bfloat16 with
a higher LoRA rank (r=64) to replicate the QLoRA capacity trade-off.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.training.lora_trainer import LoRATrainer


class QLoRATrainer(LoRATrainer):
    """
    QLoRA-configuration trainer for Apple Silicon MPS.

    On CUDA: loads base model in 4-bit NF4 (bitsandbytes) + r=64 LoRA.
    On MPS:  bitsandbytes NF4 is unavailable — loads in bfloat16 + r=64 LoRA.
    The higher rank (vs LoRA r=16) and slightly higher LR replicate the
    QLoRA adapter capacity trade-off without the quantization memory saving.
    """

    def __init__(self, config_path: str | Path = "configs/qlora_config.yaml") -> None:
        super().__init__(config_path)

    # ── Override: load model (4-bit on CUDA, bfloat16 fallback on MPS) ────────

    def _load_model(self) -> None:
        model_cfg = self.config["model"]
        logger.info(f"Loading base model for QLoRA: {model_cfg['name']}")

        use_4bit = model_cfg.get("load_in_4bit", False) and torch.cuda.is_available()

        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_cfg["name"],
                quantization_config=bnb_config,
                device_map="auto",
            )
            logger.info("4-bit NF4 model loaded ✓ (bitsandbytes CUDA path)")
        else:
            logger.info("MPS detected — bitsandbytes NF4 unavailable; loading bfloat16 (r=64 compensates)")
            self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_cfg["name"],
                torch_dtype=torch.bfloat16,
            )
            logger.info("bfloat16 model loaded ✓")

    # ── Override: train() with qlora run names and optional W&B/MLflow ────────

    def train(
        self,
        train_path: str | Path = "data/synthetic/train.jsonl",
        eval_path: str | Path = "data/synthetic/eval.jsonl",
    ) -> str:
        """QLoRA training run — same loop as LoRA, different run names."""
        import mlflow
        import wandb
        from trl import SFTTrainer

        logger.info("=== QLoRA Fine-Tuning Pipeline (r=64) ===")

        self._load_model()
        self._apply_lora()

        train_dataset = self._prepare_dataset(train_path)
        eval_dataset = self._prepare_dataset(eval_path)
        logger.info(f"Dataset: {len(train_dataset)} train / {len(eval_dataset)} eval examples")

        training_args = self._build_training_args()

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        output_dir = self.config["training"]["output_dir"]
        use_mlflow = bool(os.environ.get("MLFLOW_TRACKING_URI"))
        use_wandb = bool(os.environ.get("WANDB_API_KEY"))

        ctx = mlflow.start_run(run_name="qlora-finetune") if use_mlflow else contextlib.nullcontext()
        with ctx:
            if use_mlflow:
                mlflow.log_params(self._flatten_config())
            if use_wandb:
                wandb.init(
                    project=self.config["logging"]["wandb_project"],
                    name="qlora-finetune",
                    config=self.config,
                )

            logger.info("Starting QLoRA training...")
            trainer_stats = trainer.train()

            final_loss = trainer_stats.training_loss
            if use_mlflow:
                mlflow.log_metric("final_train_loss", final_loss)
            logger.success(f"QLoRA training complete. Final loss: {final_loss:.4f}")

            if use_wandb:
                wandb.finish()

        adapter_path = Path(output_dir) / "adapter"
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        logger.success(f"QLoRA adapter saved → {adapter_path}")

        return str(adapter_path)

    # ── Override: merge_and_export using standard PEFT ────────────────────────

    def merge_and_export(self, adapter_path: str | Path) -> str:
        """Merge QLoRA adapter into base model weights and save as bfloat16."""
        if self.model is None:
            self._load_model()
            self._apply_lora()

        merged_path = Path(self.config["training"]["output_dir"]) / "merged"
        logger.info(f"Merging QLoRA adapter into base model → {merged_path}")

        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_path))
        self.tokenizer.save_pretrained(str(merged_path))
        logger.success(f"Merged model saved → {merged_path}")
        return str(merged_path)


if __name__ == "__main__":
    trainer = QLoRATrainer("configs/qlora_config.yaml")
    adapter_path = trainer.train()
    merged_path = trainer.merge_and_export(adapter_path)
    gguf_path = trainer.export_gguf(merged_path, quantization="q4_k_m")
    print(
        f"\nQLoRA pipeline complete:\n"
        f"  Adapter: {adapter_path}\n"
        f"  Merged:  {merged_path}\n"
        f"  GGUF:    {gguf_path}"
    )
