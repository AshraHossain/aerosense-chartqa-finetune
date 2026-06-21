"""Tests for src/training/*.py — config loading and GGUF export naming.

Excluded from CI (see .github/workflows/ci.yml) because these modules import
torch/transformers/peft/trl/datasets/wandb, which CI does not install
(no GPU runner). Run locally with: pytest tests/test_training.py
"""

from __future__ import annotations

import pytest

from src.training.lora_trainer import LoRATrainer
from src.training.qlora_trainer import QLoRATrainer


@pytest.fixture
def lora_trainer() -> LoRATrainer:
    return LoRATrainer("configs/lora_config.yaml")


@pytest.fixture
def qlora_trainer() -> QLoRATrainer:
    return QLoRATrainer("configs/qlora_config.yaml")


class TestConfigLoading:
    def test_lora_config_loads_expected_keys(self, lora_trainer: LoRATrainer) -> None:
        cfg = lora_trainer.config
        assert cfg["model"]["name"] == "Qwen/Qwen2.5-3B-Instruct"
        assert cfg["lora"]["r"] == 16
        assert cfg["training"]["num_train_epochs"] == 3

    def test_qlora_config_loads_expected_keys(self, qlora_trainer: QLoRATrainer) -> None:
        cfg = qlora_trainer.config
        assert cfg["lora"]["r"] == 64


class TestAlphaRankRatio:
    """Regression test for the alpha/r scaling bug found during eval analysis:
    QLoRA's lora_alpha was 16 (with r=64, ratio 0.25) vs LoRA's 32 (with r=16,
    ratio 2.0) — an unintentional 8x weaker effective update scale that likely
    explained QLoRA tracking closest to base-model (i.e. undertrained) behavior
    on safety-critical eval cases. Fixed: qlora alpha 16 -> 128 (ratio 2.0).
    """

    def test_lora_alpha_to_rank_ratio(self, lora_trainer: LoRATrainer) -> None:
        cfg = lora_trainer.config["lora"]
        assert cfg["lora_alpha"] / cfg["r"] == 2.0

    def test_qlora_alpha_to_rank_ratio_matches_lora_convention(
        self, qlora_trainer: QLoRATrainer
    ) -> None:
        cfg = qlora_trainer.config["lora"]
        assert cfg["lora_alpha"] / cfg["r"] == 2.0


class TestGgufExportNaming:
    """Regression test for the export_gguf() filename collision bug: both
    LoRATrainer and QLoRATrainer wrote to the same hardcoded
    'aerosense-chartqa-lora-*.gguf' path, so running QLoRA training silently
    overwrote the original LoRA model's exported GGUF (and vice versa).
    Fixed via the model_export_name class attribute, overridden per subclass.
    """

    def test_lora_trainer_export_name(self, lora_trainer: LoRATrainer) -> None:
        assert lora_trainer.model_export_name == "lora"

    def test_qlora_trainer_export_name(self, qlora_trainer: QLoRATrainer) -> None:
        assert qlora_trainer.model_export_name == "qlora"

    def test_export_names_are_distinct(
        self, lora_trainer: LoRATrainer, qlora_trainer: QLoRATrainer
    ) -> None:
        assert lora_trainer.model_export_name != qlora_trainer.model_export_name

    def test_export_gguf_uses_distinct_filenames(
        self, lora_trainer: LoRATrainer, qlora_trainer: QLoRATrainer, tmp_path, monkeypatch
    ) -> None:
        # export_gguf() requires llama.cpp/convert_hf_to_gguf.py to exist before
        # it does anything else — verify the filenames it *would* produce are
        # distinct, without needing to run the actual conversion subprocess.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "llama.cpp").mkdir()
        # convert_hf_to_gguf.py existence check happens before subprocess.run,
        # so raising inside subprocess.run (mocked out) lets us inspect the
        # computed path without actually invoking llama.cpp.
        (tmp_path / "llama.cpp" / "convert_hf_to_gguf.py").touch()

        captured_paths = []

        def fake_run(cmd, check):
            captured_paths.append(cmd[4])  # value following --outfile flag
            raise RuntimeError("stop before real conversion")

        monkeypatch.setattr("subprocess.run", fake_run)

        for trainer in (lora_trainer, qlora_trainer):
            with pytest.raises(RuntimeError):
                trainer.export_gguf("dummy_merged_path")

        assert len(captured_paths) == 2
        assert captured_paths[0] != captured_paths[1]
        assert "lora-f16.gguf" in captured_paths[0]
        assert "qlora-f16.gguf" in captured_paths[1]
