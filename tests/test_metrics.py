"""Tests for pandas metrics aggregation and comparison tables."""

from exetasiz.core.models import EvalResult, TestCase
from exetasiz.metrics.metrics import compare_models, format_comparison_table, results_to_dataframe


def test_results_to_dataframe():
    tc = TestCase(id="c1", category="math", prompt="1+1", expected_answer="2")
    res = EvalResult(
        test_case_id="c1",
        model="m1",
        generated_answer="2",
        score=1.0,
        passed=True,
        evaluator="exact_match",
        latency_seconds=0.12,
        input_tokens=10,
        output_tokens=5,
    )

    df = results_to_dataframe([res], test_cases=[tc])
    assert len(df) == 1
    assert df.iloc[0]["category"] == "math"
    assert df.iloc[0]["model"] == "m1"
    assert df.iloc[0]["score"] == 1.0


def test_compare_models_aggregation():
    results = [
        EvalResult(
            test_case_id="c1",
            model="model_a",
            generated_answer="2",
            score=1.0,
            passed=True,
            evaluator="exact_match",
            latency_seconds=0.10,
            input_tokens=10,
            output_tokens=5,
        ),
        EvalResult(
            test_case_id="c2",
            model="model_a",
            generated_answer="4",
            score=1.0,
            passed=True,
            evaluator="exact_match",
            latency_seconds=0.20,
            input_tokens=10,
            output_tokens=5,
        ),
        EvalResult(
            test_case_id="c1",
            model="model_b",
            generated_answer="wrong",
            score=0.0,
            passed=False,
            evaluator="exact_match",
            latency_seconds=0.30,
            input_tokens=15,
            output_tokens=8,
        ),
        EvalResult(
            test_case_id="c2",
            model="model_b",
            generated_answer="",
            score=0.0,
            passed=False,
            evaluator="exact_match",
            latency_seconds=0.0,
            error="Connection timeout",
        ),
    ]

    summary = compare_models(results)
    assert len(summary) == 2

    # Model A
    row_a = summary[summary["model"] == "model_a"].iloc[0]
    assert row_a["total_tests"] == 2
    assert row_a["accuracy"] == 1.0
    assert row_a["pass_rate"] == 1.0
    assert row_a["failure_rate"] == 0.0
    assert row_a["total_input_tokens"] == 20
    assert row_a["total_output_tokens"] == 10

    # Model B
    row_b = summary[summary["model"] == "model_b"].iloc[0]
    assert row_b["total_tests"] == 2
    assert row_b["accuracy"] == 0.0
    assert row_b["pass_rate"] == 0.0
    assert row_b["failure_rate"] == 0.5  # 1 out of 2 had an error

    table_text = format_comparison_table(summary)
    assert "model_a" in table_text
    assert "model_b" in table_text


def test_compare_models_by_category():
    tc1 = TestCase(id="c1", category="math", prompt="1+1", expected_answer="2")
    tc2 = TestCase(id="c2", category="logic", prompt="p", expected_answer="a")

    results = [
        EvalResult(
            test_case_id="c1",
            model="m1",
            generated_answer="2",
            score=1.0,
            passed=True,
            evaluator="exact_match",
            latency_seconds=0.1,
        ),
        EvalResult(
            test_case_id="c2",
            model="m1",
            generated_answer="wrong",
            score=0.0,
            passed=False,
            evaluator="exact_match",
            latency_seconds=0.2,
        ),
    ]

    summary_cat = compare_models(results, by_category=True, test_cases=[tc1, tc2])
    assert len(summary_cat) == 2
    categories = set(summary_cat["category"])
    assert "math" in categories
    assert "logic" in categories
