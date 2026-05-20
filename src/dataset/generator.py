"""
src/dataset/generator.py

Generates synthetic aviation chart Q&A pairs using the Claude API.
Output format: Alpaca instruction-tuning JSONL.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

# ── Aviation categories and seed topics ──────────────────────────────────────

CATEGORIES: dict[str, list[str]] = {
    "chart_symbology": [
        "VFR sectional chart colors and what airspace class they represent",
        "Magenta vs blue airport symbols on sectional charts",
        "Obstruction symbols: towers, antennas, wind turbines",
        "Class B, C, D, E airspace depiction differences",
        "Military Operations Area (MOA) and Restricted Area symbols",
        "Contour lines, spot elevations, and maximum elevation figures (MEF)",
        "Navigational aid symbols: VOR, NDB, VORTAC, TACAN",
        "IFR en-route chart symbols vs VFR sectional symbols",
        "Special Use Airspace symbols and their significance",
        "Chart legend interpretation and scale",
    ],
    "approach_procedures": [
        "ILS approach components: localizer, glideslope, marker beacons",
        "Decision altitude vs minimum descent altitude",
        "Missed approach point and missed approach procedure",
        "RNAV (GPS) approach types: LPV, LNAV/VNAV, LNAV",
        "Circle-to-land minimums and visibility requirements",
        "Approach lighting systems: ALSF, MALSR, ODALS",
        "Procedure turn vs hold-in-lieu of procedure turn",
        "Initial approach fix, intermediate fix, final approach fix",
        "Alternate minimums and when alternates are required",
        "Jeppesen chart vs FAA approach plate differences",
    ],
    "airspace": [
        "Class A airspace altitude and IFR requirements",
        "Class B airspace floors, ceilings, and equipment requirements",
        "Class C airspace dimensions and communication requirements",
        "Class D airspace and tower communication requirements",
        "Class E airspace surface extensions and magenta dashed circles",
        "Class G uncontrolled airspace VFR weather minimums",
        "Temporary Flight Restrictions (TFR) types and compliance",
        "ADIZ requirements and transponder rules",
        "Mode C veil and transponder altitude reporting",
        "Special Flight Rules Areas (SFRA) procedures",
    ],
    "navigation_aids": [
        "VOR radials, bearings, and magnetic vs true north",
        "DME slant range error at low altitudes",
        "ILS localizer width and glideslope angle",
        "NDB ADF bearing vs magnetic heading relationship",
        "GPS RAIM requirements and integrity monitoring",
        "WAAS vs non-WAAS GPS approaches",
        "DME arc procedure entry and tracking",
        "Intersection identification using two VOR radials",
        "Compass locator co-located with outer marker",
        "RNP and RNAV specifications for approach procedures",
    ],
    "faa_regulations": [
        "VFR weather minimums for different airspace classes",
        "Night VFR currency requirements",
        "IFR currency: instrument approach and holding procedures",
        "Required instruments and equipment for IFR flight",
        "Fuel requirements: VFR day, VFR night, IFR",
        "ATC clearance readback requirements",
        "Pilot certificate requirements for Class B airspace",
        "RVSM airspace requirements and equipment",
        "Aircraft airworthiness and maintenance requirements",
        "Passenger briefing requirements under FAR 91.519",
    ],
    "safety_critical": [
        "Minimum safe altitude and obstacle clearance",
        "CFIT prevention and terrain awareness",
        "Wind shear and microburst recognition",
        "Spatial disorientation and unusual attitude recovery",
        "Hypoxia symptoms and oxygen requirements",
        "Fuel exhaustion vs fuel starvation",
        "Engine failure procedures and forced landing",
        "Icing conditions and airframe ice accumulation",
        "Wake turbulence separation and avoidance",
        "Emergency squawk codes and emergency declaration",
    ],
}

DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced", "expert"]

SYSTEM_PROMPT = """You are an expert aviation knowledge base generator with deep expertise in:
- FAA Aeronautical Information Manual (AIM)
- FAR/AIM regulations
- Jeppesen and FAA aeronautical chart interpretation
- Instrument procedures (TERPS criteria)
- Air traffic control procedures
- Aviation safety and DO-178C-level precision

Generate high-quality aviation Q&A pairs for fine-tuning a language model. Each Q&A must be:
1. Factually accurate per current FAA standards
2. Specific and unambiguous
3. Appropriately detailed for the difficulty level
4. Safety-conscious — never give advice that could compromise flight safety
5. Properly traceable to FAA/ICAO source documents

Format your response as valid JSON only. No preamble, no markdown, no explanation."""

GENERATION_PROMPT_TEMPLATE = """Generate {n} aviation Q&A pairs for the category: "{category}"

Topic area: {topic}
Difficulty level: {difficulty}
Safety critical: {safety_critical}

Return a JSON array of objects with this exact schema:
[
  {{
    "instruction": "<specific aviation question>",
    "input": "",
    "output": "<accurate, detailed answer — 2-4 sentences minimum>",
    "category": "{category}",
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "safety_critical": {safety_critical_lower},
    "source_reference": "<FAA AIM chapter/section, FAR part, or chart spec>"
  }}
]

Guidelines for this batch:
- Questions should test real understanding, not trivia
- Answers must be accurate enough to use in actual flight operations
- For safety_critical=true: answers must include appropriate cautions
- Vary question styles: "What is...", "How do you...", "When is...", "Why does..."
- Do NOT generate questions about specific accidents or incidents
- Do NOT generate questions that could encourage unsafe behavior"""


class DatasetGenerator:
    """Generates synthetic aviation Q&A pairs via Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        output_dir: str | Path = "data/synthetic",
        n_train: int = 450,
        n_eval: int = 50,
        batch_size: int = 5,
    ) -> None:
        self.client = anthropic.Anthropic()
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_train = n_train
        self.n_eval = n_eval
        self.batch_size = batch_size

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self) -> dict[str, list[dict[str, Any]]]:
        """Run full dataset generation. Returns train/eval split."""
        logger.info(f"Generating {self.n_train + self.n_eval} Q&A pairs...")

        all_examples: list[dict[str, Any]] = []
        target = self.n_train + self.n_eval

        category_items = list(CATEGORIES.items())
        per_category = max(1, target // len(category_items))

        for category, topics in category_items:
            logger.info(f"  Generating category: {category}")
            category_examples = self._generate_category(
                category=category,
                topics=topics,
                n=per_category,
            )
            all_examples.extend(category_examples)

            # Polite rate limiting
            time.sleep(0.5)

        # Shuffle, deduplicate, split
        random.shuffle(all_examples)
        all_examples = self._deduplicate(all_examples)

        train = all_examples[: self.n_train]
        eval_set = all_examples[self.n_train : self.n_train + self.n_eval]

        # Write to disk
        self._write_jsonl(train, self.output_dir / "train.jsonl")
        self._write_jsonl(eval_set, self.output_dir / "eval.jsonl")

        logger.success(
            f"Dataset complete — {len(train)} train, {len(eval_set)} eval examples"
        )
        return {"train": train, "eval": eval_set}

    # ── Private helpers ──────────────────────────────────────────────────────

    def _generate_category(
        self,
        category: str,
        topics: list[str],
        n: int,
    ) -> list[dict[str, Any]]:
        """Generate n examples for a given category across its topics."""
        examples: list[dict[str, Any]] = []
        batches_needed = max(1, n // self.batch_size)

        for i in range(batches_needed):
            topic = topics[i % len(topics)]
            difficulty = DIFFICULTY_LEVELS[i % len(DIFFICULTY_LEVELS)]
            safety_critical = category == "safety_critical"

            prompt = GENERATION_PROMPT_TEMPLATE.format(
                n=self.batch_size,
                category=category,
                topic=topic,
                difficulty=difficulty,
                safety_critical=str(safety_critical),
                safety_critical_lower=str(safety_critical).lower(),
            )

            try:
                batch = self._call_claude(prompt)
                # Add unique IDs
                for ex in batch:
                    ex["id"] = str(uuid.uuid4())[:8]
                examples.extend(batch)
                logger.debug(f"    Batch {i+1}/{batches_needed}: +{len(batch)} examples")
            except Exception as e:
                logger.warning(f"    Batch {i+1} failed: {e} — skipping")
                continue

        return examples

    def _call_claude(self, user_prompt: str) -> list[dict[str, Any]]:
        """Call Claude API and parse JSON response."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        content = message.content[0].text.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        parsed = json.loads(content)

        # Validate schema
        validated = []
        for item in parsed:
            if self._is_valid(item):
                validated.append(item)
            else:
                logger.warning(f"Invalid item schema, skipping: {item.get('instruction', '')[:50]}")

        return validated

    def _is_valid(self, item: dict[str, Any]) -> bool:
        """Check required fields are present and non-empty."""
        min_len = {"instruction": 10, "output": 10, "category": 3}
        return all(
            field in item and isinstance(item[field], str) and len(item[field]) >= min_len[field]
            for field in min_len
        )

    def _deduplicate(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove near-duplicate questions by exact instruction match."""
        seen: set[str] = set()
        unique = []
        for ex in examples:
            key = ex["instruction"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        logger.info(f"Deduplicated: {len(examples)} → {len(unique)} examples")
        return unique

    def _write_jsonl(self, examples: list[dict[str, Any]], path: Path) -> None:
        """Write examples to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(examples)} examples → {path}")


if __name__ == "__main__":
    generator = DatasetGenerator(n_train=450, n_eval=50)
    dataset = generator.generate()
    print(f"\nDataset ready: {len(dataset['train'])} train / {len(dataset['eval'])} eval")
