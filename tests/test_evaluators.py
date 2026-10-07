"""Tests for exact-match evaluator and placeholder stubs."""

import pytest
from exetasiz.core.models import TestCase
from exetasiz.evaluators.exact_match import ExactMatchEvaluator
from exetasiz.evaluators.placeholders import (
    HallucinationEvaluator,
    LLMJudgeEvaluator,
    SemanticSimilarityEvaluator,
)
from exetasiz.evaluators.registry import get_evaluator, list_registered_evaluators


@pytest.fixture
def base_test_case():
    return TestCase(
        id="c1",
        category="general",
        prompt="Who wrote Hamlet?",
        expected_answer="William Shakespeare",
    )


def test_exact_match_case_insensitive_and_whitespace(base_test_case):
    evaluator = ExactMatchEvaluator()

    res1 = evaluator.evaluate(base_test_case, "william shakespeare")
    assert res1.passed is True
    assert res1.score == 1.0

    res2 = evaluator.evaluate(base_test_case, "  William   Shakespeare  \n")
    assert res2.passed is True
    assert res2.score == 1.0

    res3 = evaluator.evaluate(base_test_case, "William Shakespeare.")
    assert res3.passed is True  # punctuation normalized by default


def test_exact_match_punctuation_setting():
    tc = TestCase(id="c2", category="code", prompt="cmd", expected_answer="hello-world!")
    eval_strict = ExactMatchEvaluator(normalize_punctuation=False)
    res_strict = eval_strict.evaluate(tc, "hello world")
    assert res_strict.passed is False

    eval_norm = ExactMatchEvaluator(normalize_punctuation=True)
    res_norm = eval_norm.evaluate(tc, "hello world")
    assert res_norm.passed is True


def test_exact_match_numeric_aware():
    tc = TestCase(id="math_1", category="math", prompt="What is 15% of 240?", expected_answer="36")
    evaluator = ExactMatchEvaluator(numeric_aware=True)

    # All variations should pass
    for answer in ["36", "36.0", "36.00", "  36  ", "36%"]:
        res = evaluator.evaluate(tc, answer)
        assert res.passed is True, f"Failed on: {answer}"
        assert res.score == 1.0

    # Wrong number
    res_wrong = evaluator.evaluate(tc, "42")
    assert res_wrong.passed is False
    assert res_wrong.score == 0.0


def test_evaluator_registry():
    ev = get_evaluator("exact_match", numeric_aware=True)
    assert isinstance(ev, ExactMatchEvaluator)
    assert ev.numeric_aware is True

    evals = list_registered_evaluators()
    assert "exact_match" in evals
    assert "semantic_similarity" in evals
    assert "llm_judge" in evals
    assert "hallucination" in evals


def test_evaluator_stubs_raise_not_implemented(base_test_case):
    with pytest.raises(NotImplementedError, match="SemanticSimilarityEvaluator"):
        SemanticSimilarityEvaluator().evaluate(base_test_case, "any")

    with pytest.raises(NotImplementedError, match="LLMJudgeEvaluator"):
        LLMJudgeEvaluator().evaluate(base_test_case, "any")

    with pytest.raises(NotImplementedError, match="HallucinationEvaluator"):
        HallucinationEvaluator().evaluate(base_test_case, "any")
