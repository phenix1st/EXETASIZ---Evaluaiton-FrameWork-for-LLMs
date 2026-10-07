"""Tests for standalone HTML reporting, chart generation, security escaping, and edge cases."""

from pathlib import Path
import pytest
import re

from exetasiz.core.models import (
    AggregateMetric,
    DatasetMetadata,
    EvalResult,
    LoadedRun,
    ResultsEnvelope,
    StructuredMetrics,
)
from exetasiz.reporting.charts import (
    render_accuracy_chart,
    render_grouped_category_chart,
    render_latency_chart,
)
from exetasiz.reporting.html_report import generate_html_report


@pytest.fixture
def base_run(tmp_path):
    envelope = ResultsEnvelope(
        schema_version="1.0",
        run_id="test_exp_2026-10-07",
        experiment_name="test_experiment",
        started_at="2026-10-07T12:00:00Z",
        finished_at="2026-10-07T12:00:10Z",
        duration_seconds=10.0,
        dataset=DatasetMetadata(name="test_ds", path="datasets/sample.json", num_test_cases=2),
        models=["model_alpha", "model_beta"],
        evaluators=["exact_match"],
        exetasiz_version="0.1.0",
        results=[
            EvalResult(
                test_case_id="case_1",
                model="model_alpha",
                generated_answer="Answer A",
                score=1.0,
                passed=True,
                evaluator="exact_match",
                latency_seconds=0.015,
                input_tokens=10,
                output_tokens=5,
            ),
            EvalResult(
                test_case_id="case_2",
                model="model_beta",
                generated_answer="Wrong B",
                score=0.0,
                passed=False,
                evaluator="exact_match",
                latency_seconds=0.020,
                input_tokens=10,
                output_tokens=5,
            ),
        ],
    )
    metric_alpha = AggregateMetric(
        total_evaluations=1,
        accuracy=1.0,
        pass_count=1,
        fail_count=0,
        error_count=0,
        failure_rate=0.0,
        avg_latency_s=0.015,
        p95_latency_s=0.015,
        total_input_tokens=10,
        total_output_tokens=5,
    )
    metric_beta = AggregateMetric(
        total_evaluations=1,
        accuracy=0.0,
        pass_count=0,
        fail_count=1,
        error_count=0,
        failure_rate=0.0,
        avg_latency_s=0.020,
        p95_latency_s=0.020,
        total_input_tokens=10,
        total_output_tokens=5,
    )
    overall = AggregateMetric(
        total_evaluations=2,
        accuracy=0.5,
        pass_count=1,
        fail_count=1,
        error_count=0,
        failure_rate=0.0,
        avg_latency_s=0.0175,
        p95_latency_s=0.020,
        total_input_tokens=20,
        total_output_tokens=10,
    )
    metrics = StructuredMetrics(
        overall=overall,
        by_model={"model_alpha": metric_alpha, "model_beta": metric_beta},
        by_category={"reasoning": overall},
        by_model_and_category={
            "model_alpha": {"reasoning": metric_alpha},
            "model_beta": {"reasoning": metric_beta},
        },
    )

    return LoadedRun(
        envelope=envelope,
        metrics=metrics,
        config_snapshot={"name": "test_experiment"},
        run_dir=str(tmp_path),
    )


def test_html_report_generation(base_run, tmp_path):
    report_file = tmp_path / "custom_report.html"
    generated_path = generate_html_report(base_run, output_path=report_file)

    assert generated_path.exists()
    assert generated_path.stat().st_size > 0
    content = generated_path.read_text(encoding="utf-8")

    # Contains required items
    assert "test_experiment" in content
    assert "model_alpha" in content
    assert "model_beta" in content
    assert "reasoning" in content
    assert "Exetasiz Evaluation Report" in content
    assert "Model Comparison Table" in content
    assert "Failures Explorer" in content


def test_no_external_resources_in_report(base_run, tmp_path):
    report_file = tmp_path / "offline_check.html"
    generate_html_report(base_run, output_path=report_file)
    content = report_file.read_text(encoding="utf-8")

    # Assert no external scripts, styles, or images
    assert not re.search(r'<script\s+[^>]*src=["\']https?://', content, re.IGNORECASE)
    assert not re.search(r'<link\s+[^>]*href=["\']https?://', content, re.IGNORECASE)
    assert not re.search(r'<img\s+[^>]*src=["\']https?://', content, re.IGNORECASE)
    assert not re.search(r'url\(["\']?https?://', content, re.IGNORECASE)


def test_xss_escaping_in_report(base_run, tmp_path):
    # Inject malicious payload in model answer, prompt, and error
    malicious_payload = "<script>alert(1)</script>"
    base_run.envelope.results[0].generated_answer = malicious_payload
    base_run.envelope.results[1].error = f"Crash {malicious_payload}"

    report_file = tmp_path / "xss_test.html"
    generate_html_report(base_run, output_path=report_file)
    content = report_file.read_text(encoding="utf-8")

    # Payload must NOT be present unescaped
    assert "<script>alert(1)</script>" not in content
    # Must be escaped
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content


def test_edge_case_single_model(base_run, tmp_path):
    base_run.envelope.models = ["solo_model"]
    base_run.metrics.by_model = {"solo_model": base_run.metrics.by_model["model_alpha"]}
    base_run.metrics.by_model_and_category = {"solo_model": {"reasoning": base_run.metrics.by_model["solo_model"]}}

    report_file = tmp_path / "single_model.html"
    generated_path = generate_html_report(base_run, output_path=report_file)
    assert generated_path.exists()
    content = generated_path.read_text(encoding="utf-8")
    assert "solo_model" in content


def test_edge_case_no_failures(base_run, tmp_path):
    # Make all results passed
    for r in base_run.envelope.results:
        r.passed = True
        r.score = 1.0
        r.error = None

    report_file = tmp_path / "no_failures.html"
    generate_html_report(base_run, output_path=report_file)
    content = report_file.read_text(encoding="utf-8")
    assert "All evaluations passed with zero errors or failures" in content


def test_edge_case_null_latency_and_tokens(base_run, tmp_path):
    for r in base_run.envelope.results:
        r.latency_seconds = 0.0
        r.input_tokens = None
        r.output_tokens = None
    for m in base_run.metrics.by_model.values():
        m.avg_latency_s = None
        m.p95_latency_s = None
        m.total_input_tokens = None
        m.total_output_tokens = None

    report_file = tmp_path / "null_metrics.html"
    generate_html_report(base_run, output_path=report_file)
    content = report_file.read_text(encoding="utf-8")
    assert content.count("-") > 0


def test_edge_case_many_models(base_run, tmp_path):
    # Test scalability with 12 models
    many_models = [f"model_{i}" for i in range(12)]
    base_run.envelope.models = many_models
    base_run.metrics.by_model = {
        m: AggregateMetric(
            total_evaluations=1,
            accuracy=0.8,
            pass_count=1,
            fail_count=0,
            error_count=0,
            failure_rate=0.0,
            avg_latency_s=0.01,
            p95_latency_s=0.01,
        )
        for m in many_models
    }
    base_run.metrics.by_model_and_category = {
        m: {"reasoning": base_run.metrics.by_model[m]}
        for m in many_models
    }

    report_file = tmp_path / "many_models.html"
    generate_html_report(base_run, output_path=report_file)
    assert report_file.exists()
