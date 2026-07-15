"""Tests for src/dataset/generator.py — schema validation and dedup logic.

These exercise pure data-processing methods on DatasetGenerator without
calling the Claude API (no network calls; ANTHROPIC_API_KEY not required
since anthropic.Anthropic() only needs a key at call time, not construction).
"""

from __future__ import annotations

import json

import pytest

from src.dataset.generator import CATEGORIES, DIFFICULTY_LEVELS, DatasetGenerator


@pytest.fixture
def generator(tmp_path) -> DatasetGenerator:
    return DatasetGenerator(output_dir=tmp_path, n_train=10, n_eval=2, batch_size=2)


VALID_ITEM = {
    "instruction": "What is a Vmc demonstration?",
    "input": "",
    "output": "Vmc demonstration shows minimum control speed in flight.",
    "category": "safety_critical",
    "topic": "Vmc",
    "difficulty": "advanced",
    "safety_critical": True,
    "source_reference": "FAA-H-8083-3C",
}


class TestIsValid:
    def test_accepts_well_formed_item(self, generator: DatasetGenerator) -> None:
        assert generator._is_valid(VALID_ITEM) is True

    @pytest.mark.parametrize("field", ["instruction", "output", "category", "source_reference"])
    def test_rejects_missing_required_field(self, generator: DatasetGenerator, field: str) -> None:
        item = dict(VALID_ITEM)
        del item[field]
        assert generator._is_valid(item) is False

    def test_rejects_too_short_instruction(self, generator: DatasetGenerator) -> None:
        item = dict(VALID_ITEM, instruction="short")
        assert generator._is_valid(item) is False

    def test_rejects_too_short_source_reference(self, generator: DatasetGenerator) -> None:
        item = dict(VALID_ITEM, source_reference="FAR")
        assert generator._is_valid(item) is False

    def test_rejects_non_string_field(self, generator: DatasetGenerator) -> None:
        item = dict(VALID_ITEM, instruction=12345)
        assert generator._is_valid(item) is False

    def test_accepts_minimum_length_boundary(self, generator: DatasetGenerator) -> None:
        # instruction min_len is 10 — exactly 10 chars should pass
        item = dict(VALID_ITEM, instruction="1234567890")
        assert generator._is_valid(item) is True

    def test_rejects_one_below_minimum_length(self, generator: DatasetGenerator) -> None:
        item = dict(VALID_ITEM, instruction="123456789")
        assert generator._is_valid(item) is False


class TestDeduplicate:
    def test_removes_exact_duplicate_instructions(self, generator: DatasetGenerator) -> None:
        examples = [dict(VALID_ITEM), dict(VALID_ITEM)]
        result = generator._deduplicate(examples)
        assert len(result) == 1

    def test_is_case_and_whitespace_insensitive(self, generator: DatasetGenerator) -> None:
        a = dict(VALID_ITEM, instruction="What is a Vmc demonstration?")
        b = dict(VALID_ITEM, instruction="  WHAT IS A VMC DEMONSTRATION?  ")
        result = generator._deduplicate([a, b])
        assert len(result) == 1

    def test_keeps_distinct_instructions(self, generator: DatasetGenerator) -> None:
        a = dict(VALID_ITEM, instruction="What is a Vmc demonstration?")
        b = dict(VALID_ITEM, instruction="What is the standard ILS glideslope angle?")
        result = generator._deduplicate([a, b])
        assert len(result) == 2

    def test_empty_input_returns_empty(self, generator: DatasetGenerator) -> None:
        assert generator._deduplicate([]) == []

    def test_preserves_first_occurrence_order(self, generator: DatasetGenerator) -> None:
        a = dict(VALID_ITEM, instruction="First question here", output="first")
        b = dict(VALID_ITEM, instruction="first question here", output="second")
        result = generator._deduplicate([a, b])
        assert len(result) == 1
        assert result[0]["output"] == "first"


class TestWriteJsonl:
    def test_writes_one_json_object_per_line(self, generator: DatasetGenerator, tmp_path) -> None:
        examples = [dict(VALID_ITEM), dict(VALID_ITEM, instruction="Different question text")]
        out_path = tmp_path / "out.jsonl"
        generator._write_jsonl(examples, out_path)

        lines = out_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "instruction" in parsed

    def test_writes_empty_file_for_no_examples(self, generator: DatasetGenerator, tmp_path) -> None:
        out_path = tmp_path / "empty.jsonl"
        generator._write_jsonl([], out_path)
        assert out_path.read_text(encoding="utf-8") == ""


class TestCategoriesAndDifficultyConstants:
    def test_six_categories_defined(self) -> None:
        assert len(CATEGORIES) == 6

    def test_every_category_has_topics(self) -> None:
        for category, topics in CATEGORIES.items():
            assert len(topics) > 0, f"{category} has no seed topics"

    def test_difficulty_levels_are_ordered_set(self) -> None:
        assert DIFFICULTY_LEVELS == ["beginner", "intermediate", "advanced", "expert"]


class TestDatasetGeneratorConstruction:
    def test_constructs_without_api_key(self, monkeypatch, tmp_path) -> None:
        # anthropic.Anthropic() only requires a key at call time, not construction —
        # generator instantiation must not require ANTHROPIC_API_KEY to be set.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        gen = DatasetGenerator(output_dir=tmp_path)
        assert gen.n_train == 450
        assert gen.n_eval == 50

    def test_creates_output_dir(self, tmp_path) -> None:
        target = tmp_path / "nested" / "synthetic"
        DatasetGenerator(output_dir=target)
        assert target.exists()
