"""
scripts/topup_airspace.py

Generates the missing airspace examples (category name was 8 chars, below
the old 10-char minimum in _is_valid — now fixed). Merges with existing
train.jsonl, deduplicates, and re-splits into 450 train / 50 eval.
"""

from __future__ import annotations

import json
import random
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic  # noqa: E402
from loguru import logger  # noqa: E402

from src.dataset.generator import (  # noqa: E402
    CATEGORIES,
    DIFFICULTY_LEVELS,
    GENERATION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 5
N_BATCHES = 16  # 16 × 5 = 80 airspace examples
TRAIN_PATH = Path("data/synthetic/train.jsonl")
EVAL_PATH  = Path("data/synthetic/eval.jsonl")
N_TRAIN    = 450
N_EVAL     = 50


def call_claude(client: anthropic.Anthropic, prompt: str) -> list[dict]:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    block = msg.content[0]
    assert isinstance(block, anthropic.types.TextBlock)
    content = block.text.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])
    return json.loads(content)


def is_valid(item: dict) -> bool:
    min_len = {"instruction": 10, "output": 10, "category": 3}
    return all(
        field in item and isinstance(item[field], str) and len(item[field]) >= min_len[field]
        for field in min_len
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]


def write_jsonl(examples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(examples)} examples → {path}")


def main() -> None:
    client = anthropic.Anthropic()
    topics = CATEGORIES["airspace"]
    new_examples: list[dict] = []

    logger.info(f"Generating {N_BATCHES} airspace batches ({N_BATCHES * BATCH_SIZE} examples)...")
    for i in range(N_BATCHES):
        topic = topics[i % len(topics)]
        difficulty = DIFFICULTY_LEVELS[i % len(DIFFICULTY_LEVELS)]
        prompt = GENERATION_PROMPT_TEMPLATE.format(
            n=BATCH_SIZE,
            category="airspace",
            topic=topic,
            difficulty=difficulty,
            safety_critical="False",
            safety_critical_lower="false",
        )
        try:
            raw = call_claude(client, prompt)
            batch = [ex for ex in raw if is_valid(ex)]
            for ex in batch:
                ex.setdefault("id", str(uuid.uuid4())[:8])
            new_examples.extend(batch)
            logger.debug(f"  Batch {i+1}/{N_BATCHES}: +{len(batch)} examples")
        except Exception as e:
            logger.warning(f"  Batch {i+1} failed: {e} — skipping")

    logger.info(f"Airspace examples generated: {len(new_examples)}")

    # Merge with existing train set
    existing = read_jsonl(TRAIN_PATH)
    logger.info(f"Existing train examples: {len(existing)}")

    combined = existing + new_examples
    random.shuffle(combined)

    # Deduplicate by instruction
    seen: set[str] = set()
    unique = []
    for ex in combined:
        key = ex["instruction"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    logger.info(f"Deduplicated: {len(combined)} → {len(unique)} examples")

    train = unique[:N_TRAIN]
    eval_set = unique[N_TRAIN:N_TRAIN + N_EVAL]

    write_jsonl(train, TRAIN_PATH)
    write_jsonl(eval_set, EVAL_PATH)

    logger.success(
        f"Dataset ready: {len(train)} train / {len(eval_set)} eval "
        f"({'OK' if len(eval_set) == N_EVAL else 'SHORT — need more examples'})"
    )


if __name__ == "__main__":
    main()
