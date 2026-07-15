"""Tests for src/evaluation/judge.py — eval-set loading and result aggregation.

Pure data-processing logic only; no Claude API calls (anthropic.Anthropic()
does not require ANTHROPIC_API_KEY at construction time).
"""

from __future__ import annotations

import json

import pytest

from src.evaluation.judge import EvalResult, EvalSummary, LLMJudge


@pytest.fixture
def judge(tmp_path) -> LLMJudge:
    return LLMJudge(output_dir=tmp_path)


def make_result(
    domain_accuracy: float = 5.0,
    hallucination_score: float = 0.5,
    safety_refusal_score: float = 0.5,
    error: str = "",
) -> EvalResult:
    return EvalResult(
        question="Q",
        reference="R",
        response="A",
        model_name="test-model",
        category="airspace",
        domain_accuracy=domain_accuracy,
        hallucination_score=hallucination_score,
        safety_refusal_score=safety_refusal_score,
        error=error,
    )


class TestLoadEvalSet:
    def test_loads_all_lines(self, judge: LLMJudge, tmp_path) -> None:
        path = tmp_path / "eval.jsonl"
        rows = [{"instruction": "Q1", "output": "A1"}, {"instruction": "Q2", "output": "A2"}]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        loaded = judge._load_eval_set(path)
        assert len(loaded) == 2
        assert loaded[0]["instruction"] == "Q1"

    def test_handles_trailing_newline(self, judge: LLMJudge, tmp_path) -> None:
        path = tmp_path / "eval.jsonl"
        path.write_text('{"instruction": "Q1", "output": "A1"}\n\n', encoding="utf-8")
        # A blank trailing line would raise json.JSONDecodeError if not guarded —
        # this only passes if the loader skips/strips empty lines correctly.
        with pytest.raises(json.JSONDecodeError):
            judge._load_eval_set(path)


class TestSummarize:
    def test_averages_scores_across_valid_results(self, judge: LLMJudge) -> None:
        results = [
            make_result(domain_accuracy=8.0, hallucination_score=1.0, safety_refusal_score=1.0),
            make_result(domain_accuracy=4.0, hallucination_score=0.0, safety_refusal_score=0.0),
        ]
        summary = judge._summarize("test-model", results)
        assert summary.n_examples == 2
        assert summary.mean_domain_accuracy == 6.0
        assert summary.mean_hallucination_score == 0.5
        assert summary.mean_safety_refusal_score == 0.5

    def test_excludes_errored_results_from_aggregate(self, judge: LLMJudge) -> None:
        results = [
            make_result(domain_accuracy=10.0),
            make_result(domain_accuracy=0.0, error="judge scoring failed"),
        ]
        summary = judge._summarize("test-model", results)
        assert summary.n_examples == 1
        assert summary.mean_domain_accuracy == 10.0

    def test_empty_results_produce_zeroed_summary(self, judge: LLMJudge) -> None:
        summary = judge._summarize("test-model", [])
        assert summary.n_examples == 0
        assert summary.mean_domain_accuracy == 0
        assert summary.mean_hallucination_score == 0
        assert summary.mean_safety_refusal_score == 0

    def test_std_requires_at_least_two_valid_results(self, judge: LLMJudge) -> None:
        summary = judge._summarize("test-model", [make_result(domain_accuracy=5.0)])
        assert summary.std_domain_accuracy == 0

    def test_all_errored_produces_zeroed_summary(self, judge: LLMJudge) -> None:
        results = [make_result(error="failed"), make_result(error="failed")]
        summary = judge._summarize("test-model", results)
        assert summary.n_examples == 0
        assert summary.mean_domain_accuracy == 0


class TestEvalSummaryAsDict:
    def test_rounds_to_three_decimals(self) -> None:
        summary = EvalSummary(
            model_name="qlora",
            n_examples=50,
            mean_domain_accuracy=3.4567,
            mean_hallucination_score=0.23456,
            mean_safety_refusal_score=0.41234,
            std_domain_accuracy=1.34123,
        )
        d = summary.as_dict()
        assert d["mean_domain_accuracy"] == 3.457
        assert d["mean_hallucination_score"] == 0.235
        assert d["mean_safety_refusal_score"] == 0.412
        assert d["std_domain_accuracy"] == 1.341

    def test_excludes_results_list_from_dict(self) -> None:
        summary = EvalSummary(
            model_name="qlora",
            n_examples=1,
            mean_domain_accuracy=5.0,
            mean_hallucination_score=0.5,
            mean_safety_refusal_score=0.5,
            std_domain_accuracy=0.0,
            results=[make_result()],
        )
        assert "results" not in summary.as_dict()


class TestScoreEmptyResponse:
    def test_empty_response_short_circuits_with_error(self, judge: LLMJudge) -> None:
        result = judge._score(
            question="Q",
            reference="R",
            response="",
            model_name="test-model",
            category="airspace",
        )
        assert result.error == "Empty response from model"
        assert result.domain_accuracy == 0.0
