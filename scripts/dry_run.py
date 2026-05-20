"""
scripts/dry_run.py

Dry-run validation: generate 10 Q&A pairs (2 batches × 5) across two
aviation categories and validate the Alpaca JSONL schema before the
full 500-example run.

Usage:
    python scripts/dry_run.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic
from loguru import logger

from src.dataset.generator import (
    CATEGORIES,
    DIFFICULTY_LEVELS,
    GENERATION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)

# ── Config ─────────────────────────────────────────────────────────────────────

DRY_RUN_MODEL = "claude-sonnet-4-6"  # generator.py uses claude-sonnet-4-20250514 — update it too
DRY_RUN_OUTPUT = Path("data/synthetic/dry_run.jsonl")
BATCH_SIZE = 5
N_BATCHES = 2  # 2 × 5 = 10 examples total

REQUIRED_FIELDS = {
    "instruction": str,
    "input": str,
    "output": str,
    "category": str,
    "topic": str,
    "difficulty": str,
    "safety_critical": bool,
    "source_reference": str,
}

# ── Helpers ────────────────────────────────────────────────────────────────────


def call_claude(client: anthropic.Anthropic, prompt: str) -> list[dict]:
    message = client.messages.create(
        model=DRY_RUN_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    content = message.content[0].text.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])
    return json.loads(content)


MIN_LEN: dict[str, int] = {
    "instruction": 10,
    "input": 0,       # always "" in Alpaca format
    "output": 10,
    "category": 3,
    "topic": 3,
    "difficulty": 3,  # shortest value is "expert" (6 chars)
    "source_reference": 5,
}


def validate_item(item: dict, idx: int) -> list[str]:
    """Return list of validation errors; empty = pass."""
    errors: list[str] = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in item:
            errors.append(f"[{idx}] missing field: '{field}'")
        elif not isinstance(item[field], expected_type):
            errors.append(
                f"[{idx}] field '{field}' expected {expected_type.__name__}, "
                f"got {type(item[field]).__name__}"
            )
        elif expected_type is str and len(item[field]) < MIN_LEN.get(field, 10):
            errors.append(f"[{idx}] field '{field}' too short (< {MIN_LEN.get(field, 10)} chars)")
    if item.get("difficulty") not in DIFFICULTY_LEVELS:
        errors.append(f"[{idx}] invalid difficulty: {item.get('difficulty')!r}")
    if item.get("output") and len(item["output"].split()) < 20:
        errors.append(f"[{idx}] output suspiciously short (< 20 words)")
    return errors


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info(f"Dry run — model: {DRY_RUN_MODEL}, target: {BATCH_SIZE * N_BATCHES} examples")
    logger.warning(
        "NOTE: generator.py uses 'claude-sonnet-4-20250514' which is not a current "
        "model ID. Update DatasetGenerator(model=...) to 'claude-sonnet-4-6'."
    )

    client = anthropic.Anthropic()
    categories = list(CATEGORIES.items())
    all_items: list[dict] = []
    all_errors: list[str] = []

    for batch_idx in range(N_BATCHES):
        category, topics = categories[batch_idx % len(categories)]
        topic = topics[batch_idx % len(topics)]
        difficulty = DIFFICULTY_LEVELS[batch_idx % len(DIFFICULTY_LEVELS)]
        safety_critical = category == "safety_critical"

        prompt = GENERATION_PROMPT_TEMPLATE.format(
            n=BATCH_SIZE,
            category=category,
            topic=topic,
            difficulty=difficulty,
            safety_critical=str(safety_critical),
            safety_critical_lower=str(safety_critical).lower(),
        )

        logger.info(f"Batch {batch_idx + 1}/{N_BATCHES}: category={category!r}, difficulty={difficulty}")

        try:
            raw = call_claude(client, prompt)
        except Exception as exc:
            logger.error(f"API call failed: {exc}")
            sys.exit(1)

        for item in raw:
            item.setdefault("id", str(uuid.uuid4())[:8])
            errors = validate_item(item, len(all_items))
            if errors:
                all_errors.extend(errors)
                logger.warning(f"  Schema errors: {errors}")
            else:
                logger.debug(f"  OK: {item['instruction'][:70]}")
            all_items.append(item)

    # ── Write output ──────────────────────────────────────────────────────────

    DRY_RUN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DRY_RUN_OUTPUT, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ── Summary report ────────────────────────────────────────────────────────

    n_total = len(all_items)
    n_errors = len(all_errors)
    print("\n" + "=" * 60)
    print(f"DRY RUN REPORT")
    print("=" * 60)
    print(f"  Examples generated : {n_total}")
    print(f"  Schema errors      : {n_errors}")
    print(f"  Output             : {DRY_RUN_OUTPUT}")
    print()

    if n_errors:
        print("ERRORS:")
        for e in all_errors:
            print(f"  {e}")
        print()

    print("SAMPLE (first item):")
    if all_items:
        sample = all_items[0]
        print(f"  id               : {sample.get('id')}")
        print(f"  category         : {sample.get('category')}")
        print(f"  difficulty       : {sample.get('difficulty')}")
        print(f"  safety_critical  : {sample.get('safety_critical')}")
        print(f"  source_reference : {sample.get('source_reference')}")
        print(f"  instruction      : {sample.get('instruction')[:100]}")
        print(f"  output (preview) : {str(sample.get('output', ''))[:150]}...")
    print("=" * 60)

    if n_errors == 0:
        print("\nSchema validation PASSED. Ready for full generation.")
    else:
        print(f"\nSchema validation FAILED ({n_errors} errors). Fix issues before full run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
