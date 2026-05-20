"""
src/training/qlora_trainer.py

Fine-tunes Qwen2.5-3B with QLoRA (4-bit NF4 quantization) via Unsloth + bitsandbytes.
Designed for memory-constrained environments. Extends LoRATrainer with quantization overrides.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import wandb
from loguru import logger
from unsloth import FastLanguageModel

from src.training.lora_trainer import LoRATrainer


class QLoRATrainer(LoRATrainer):
    """
    QLoRA variant — loads base model in 4-bit NF4 quantization.
    Everything else (adapter application, dataset prep, training loop) is inherited.

    Key differences vs LoRA:
    - load_in_4bit=True with bitsandbytes NF4 quantization
    - Higher LoRA rank (r=64) to compensate for quantization noise
    - Lower alpha (16) for stable gradient flow
    - Slightly higher learning rate typical for QLoRA
    """

    def __init__(self, config_path: str | Path = "configs/qlora_config.yaml") -> None:
        super().__init__(config_path)

    # ── Override: load model in 4-bit ────────────────────────────────────────

    def _load_model(self) -> None:
        model_cfg = self.config["model"]
        logger.info(f"Loading base model in 4-bit NF4: {model_cfg['name']}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg["name"],
            max_seq_length=self.config["training"]["max_seq_length"],
            dtype=None,
            load_in_4bit=True,                                    # ← QLoRA: 4-bit
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"4-bit model loaded ✓  |  Parameters: {total_params / 1e9:.2f}B  "
            f"|  Estimated VRAM: ~{total_params * 0.5 / 1e9:.1f}GB (NF4)"
        )

    # ── Override: train() to use qlora run names ─────────────────────────────

    def train(
        self,
        train_path: str | Path = "data/synthetic/train.jsonl",
        eval_path: str | Path = "data/synthetic/eval.jsonl",
    ) -> str:
        """QLoRA training run — overrides run name for W&B / MLflow tracking."""
        logger.info("=== QLoRA Fine-Tuning Pipeline (4-bit NF4) ===")

        self._load_model()
        self._apply_lora()

        from datasets import Dataset

        train_dataset = self._prepare_dataset(train_path)
        eval_dataset = self._prepare_dataset(eval_path)

        logger.info(
            f"Dataset: {len(train_dataset)} train / {len(eval_dataset)} eval examples"
        )

        from trl import SFTTrainer

        training_args = self._build_training_args()

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config["training"]["max_seq_length"],
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        output_dir = self.config["training"]["output_dir"]

        with mlflow.start_run(run_name="qlora-finetune"):
            mlflow.log_params(self._flatten_config())
            wandb.init(
                project=self.config["logging"]["wandb_project"],
                name="qlora-finetune",
                config=self.config,
            )

            logger.info("Starting QLoRA training...")
            trainer_stats = trainer.train()

            final_loss = trainer_stats.training_loss
            mlflow.log_metric("final_train_loss", final_loss)
            logger.success(f"QLoRA training complete. Final loss: {final_loss:.4f}")

            wandb.finish()

        adapter_path = Path(output_dir) / "adapter"
        self.model.save_pretrained(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        logger.success(f"QLoRA adapter saved → {adapter_path}")

        return str(adapter_path)

    def merge_and_export(self, adapter_path: str | Path) -> str:
        """
        QLoRA merge: dequantize + merge → save as fp16.
        Note: merged model will be larger than 4-bit checkpoint.
        """
        if self.model is None:
            self._load_model()
            self._apply_lora()

        merged_path = Path(self.config["training"]["output_dir"]) / "merged"
        logger.info(
            f"Dequantizing and merging QLoRA adapter → {merged_path}\n"
            "Note: merged model saved as fp16 (~6GB for 3B params)"
        )

        self.model.save_pretrained_merged(
            str(merged_path),
            self.tokenizer,
            save_method="merged_16bit",
        )
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
