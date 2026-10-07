"""Tests for core domain schemas."""

import pytest
from pydantic import ValidationError

from exetasiz.core.models import Dataset, EvalResult, RunResult, TestCase


def test_test_case_valid():
    tc = TestCase(
        id="case_1",
        category="math",
        prompt="1+1?",
        expected_answer="2",
        context="simple addition",
        metadata={"difficulty": "easy"},
    )
    assert tc.id == "case_1"
    assert tc.category == "math"
    assert tc.expected_answer == "2"
    assert tc.metadata["difficulty"] == "easy"


def test_test_case_missing_required():
    with pytest.raises(ValidationError):
        TestCase(id="c1", prompt="hello", expected_answer="hi")  # missing category


def test_eval_result_valid():
    er = EvalResult(
        test_case_id="case_1",
        model="mock",
        generated_answer="2",
        score=1.0,
        passed=True,
        evaluator="exact_match",
        latency_seconds=0.05,
    )
    assert er.passed is True
    assert er.score == 1.0
    assert er.error is None


def test_eval_result_score_bounds():
    with pytest.raises(ValidationError):
        EvalResult(
            test_case_id="case_1",
            model="mock",
            generated_answer="2",
            score=1.5,  # score must be <= 1.0
            passed=True,
            evaluator="exact_match",
            latency_seconds=0.05,
        )


def test_dataset_valid():
    tc = TestCase(id="c1", category="c", prompt="p", expected_answer="a")
    ds = Dataset(name="ds1", test_cases=[tc])
    assert len(ds.test_cases) == 1


def test_run_result():
    tc = TestCase(id="c1", category="c", prompt="p", expected_answer="a")
    er = EvalResult(
        test_case_id="c1",
        model="mock",
        generated_answer="a",
        score=1.0,
        passed=True,
        evaluator="exact_match",
        latency_seconds=0.01,
    )
    rr = RunResult(
        experiment_name="test_exp",
        timestamp="2026-10-07T00:00:00",
        results=[er],
        config_snapshot={"k": "v"},
    )
    assert rr.experiment_name == "test_exp"
    assert len(rr.results) == 1
