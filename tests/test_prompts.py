"""Tests for src/prompts.py — the shared Alpaca prompt template."""

from __future__ import annotations

from src.prompts import ALPACA_PROMPT, format_for_inference, format_for_training


class TestFormatForTraining:
    def test_fills_all_three_slots(self) -> None:
        result = format_for_training("What is Vmc?", "", "Vmc is minimum control speed.")
        assert "What is Vmc?" in result
        assert "Vmc is minimum control speed." in result

    def test_includes_context_when_provided(self) -> None:
        result = format_for_training("Q", "some context", "A")
        assert "some context" in result

    def test_response_is_not_blank(self) -> None:
        result = format_for_training("Q", "", "the answer")
        # Unlike format_for_inference, the Response section must be filled in
        response_section = result.split("### Response:")[1].strip()
        assert response_section == "the answer"


class TestFormatForInference:
    def test_leaves_response_blank(self) -> None:
        result = format_for_inference("What is Vmc?")
        response_section = result.split("### Response:")[1].strip()
        assert response_section == ""

    def test_default_context_is_empty(self) -> None:
        result = format_for_inference("Q")
        input_section = result.split("### Input:")[1].split("### Response:")[0].strip()
        assert input_section == ""

    def test_includes_instruction(self) -> None:
        result = format_for_inference("How does Vmc change with altitude?")
        assert "How does Vmc change with altitude?" in result


class TestPromptConsistency:
    def test_training_and_inference_share_same_template(self) -> None:
        # Both helpers must derive from the same ALPACA_PROMPT — this is the
        # whole point of centralizing it (prevents train/inference format drift).
        train_result = format_for_training("Q", "C", "R")
        inference_result = format_for_inference("Q", "C")
        train_prefix = train_result.split("### Response:")[0]
        inference_prefix = inference_result.split("### Response:")[0]
        assert train_prefix == inference_prefix

    def test_template_has_expected_sections(self) -> None:
        assert "### Instruction:" in ALPACA_PROMPT
        assert "### Input:" in ALPACA_PROMPT
        assert "### Response:" in ALPACA_PROMPT
