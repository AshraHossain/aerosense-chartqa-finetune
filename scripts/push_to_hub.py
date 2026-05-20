"""
scripts/push_to_hub.py

Push fine-tuned adapters and GGUF models to Hugging Face Hub.

Requires HF_TOKEN in environment (set in .env or shell).

Usage:
  python scripts/push_to_hub.py
  python scripts/push_to_hub.py --org my-org
  python scripts/push_to_hub.py --skip-gguf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi
from loguru import logger


_CONFIGS = {
    "lora":  "configs/lora_config.yaml",
    "qlora": "configs/qlora_config.yaml",
}

_REPO_NAMES = {
    "lora":  "aerosense-chartqa-lora",
    "qlora": "aerosense-chartqa-qlora",
}


def _load_output_dir(config_path: str) -> Path:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["training"]["output_dir"])


def push_adapter(
    variant: str,
    org: str | None,
    api: HfApi,
    token: str,
) -> str:
    """Push adapter weights for one variant (lora or qlora). Returns repo id."""
    output_dir = _load_output_dir(_CONFIGS[variant])
    adapter_path = output_dir / "adapter"

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_path}. "
            f"Run --step {variant} first."
        )

    repo_name = _REPO_NAMES[variant]
    repo_id = f"{org}/{repo_name}" if org else repo_name

    logger.info(f"Pushing {variant} adapter → {repo_id}")
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        token=token,
        commit_message=f"Upload {variant} adapter weights",
    )
    logger.success(f"{variant} adapter → https://huggingface.co/{repo_id}")
    return repo_id


def push_gguf(
    variant: str,
    org: str | None,
    api: HfApi,
    token: str,
) -> None:
    """Push GGUF file for one variant if it exists."""
    models_dir = Path("models")
    gguf_files = list(models_dir.glob(f"aerosense-chartqa-{variant}-*.gguf"))
    if not gguf_files:
        logger.warning(f"No GGUF found for {variant} in {models_dir}/ — skipping")
        return

    repo_name = _REPO_NAMES[variant]
    repo_id = f"{org}/{repo_name}" if org else repo_name

    for gguf_path in gguf_files:
        logger.info(f"Uploading {gguf_path.name} → {repo_id}")
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=repo_id,
            token=token,
            commit_message=f"Upload {gguf_path.name}",
        )
        logger.success(f"GGUF uploaded → https://huggingface.co/{repo_id}/{gguf_path.name}")


def push_all(org: str | None = None, skip_gguf: bool = False) -> None:
    """Push all adapters (and optionally GGUFs) to Hugging Face Hub."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not set. Add it to .env or export it in your shell."
        )

    api = HfApi()

    for variant in ("lora", "qlora"):
        try:
            push_adapter(variant, org=org, api=api, token=token)
            if not skip_gguf:
                push_gguf(variant, org=org, api=api, token=token)
        except FileNotFoundError as e:
            logger.warning(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push models to Hugging Face Hub")
    parser.add_argument("--org", default=None, help="HF org or username (default: token owner)")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF uploads")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    push_all(org=args.org, skip_gguf=args.skip_gguf)
