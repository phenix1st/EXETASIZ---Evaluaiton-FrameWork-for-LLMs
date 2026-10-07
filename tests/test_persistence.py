"""Tests for structured result persistence, load_run, and atomic writes."""

import json
from pathlib import Path
import pytest

from exetasiz.core.models import (
    AggregateMetric,
    DatasetMetadata,
    EvalResult,
    LoadedRun,
    ResultsEnvelope,
    StructuredMetrics,
)
from exetasiz.core.persistence import (
    RunLoadError,
    UnsupportedSchemaVersionError,
    atomic_write_file,
    atomic_write_json,
    list_saved_runs,
    load_run,
    save_run,
)


@pytest.fixture
def sample_envelope():
    return ResultsEnvelope(
        schema_version="1.0",
        run_id="test_run_001",
        experiment_name="test_exp",
        started_at="2026-10-07T20:00:00Z",
        finished_at="2026-10-07T20:00:05Z",
        duration_seconds=5.0,
        dataset=DatasetMetadata(name="test_ds", path="datasets/test.json", num_test_cases=2),
        models=["m1", "m2"],
        evaluators=["exact_match"],
        exetasiz_version="0.1.0",
        results=[
            EvalResult(
                test_case_id="c1",
                model="m1",
                generated_answer="ans1",
                score=1.0,
                passed=True,
                evaluator="exact_match",
                latency_seconds=0.01,
            )
        ],
    )


@pytest.fixture
def sample_metrics():
    agg = AggregateMetric(
        total_evaluations=1,
        accuracy=1.0,
        pass_count=1,
        fail_count=0,
        error_count=0,
        failure_rate=0.0,
        avg_latency_s=0.01,
        p95_latency_s=0.01,
        total_input_tokens=10,
        total_output_tokens=5,
    )
    return StructuredMetrics(
        overall=agg,
        by_model={"m1": agg},
        by_category={"general": agg},
        by_model_and_category={"m1": {"general": agg}},
    )


def test_save_and_load_run_valid(tmp_path, sample_envelope, sample_metrics):
    run_dir = tmp_path / "valid_run"
    config_snap = {"name": "test_exp", "models": ["m1", "m2"]}

    saved_path = save_run(run_dir, sample_envelope, sample_metrics, config_snap)
    assert saved_path.exists()
    assert (saved_path / "results.json").exists()
    assert (saved_path / "metrics.json").exists()
    assert (saved_path / "config.yaml").exists()

    loaded = load_run(run_dir)
    assert isinstance(loaded, LoadedRun)
    assert loaded.envelope.experiment_name == "test_exp"
    assert loaded.envelope.schema_version == "1.0"
    assert loaded.metrics.overall.accuracy == 1.0
    assert loaded.config_snapshot["name"] == "test_exp"


def test_load_run_missing_dir(tmp_path):
    with pytest.raises(RunLoadError, match="Run directory not found"):
        load_run(tmp_path / "nonexistent")


def test_load_run_missing_files(tmp_path):
    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()
    with pytest.raises(RunLoadError, match="Missing required file 'results.json'"):
        load_run(empty_dir)


def test_load_run_corrupt_json(tmp_path, sample_envelope, sample_metrics):
    run_dir = tmp_path / "corrupt_run"
    save_run(run_dir, sample_envelope, sample_metrics, {})
    (run_dir / "results.json").write_text("{bad json syntax", encoding="utf-8")

    with pytest.raises(RunLoadError, match="Corrupt or invalid JSON"):
        load_run(run_dir)


def test_load_run_unsupported_schema_version(tmp_path, sample_envelope, sample_metrics):
    run_dir = tmp_path / "old_schema_run"
    save_run(run_dir, sample_envelope, sample_metrics, {})

    results_data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    results_data["schema_version"] = "99.0"
    (run_dir / "results.json").write_text(json.dumps(results_data), encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersionError, match="Unsupported schema_version '99.0'"):
        load_run(run_dir)


def test_atomic_write(tmp_path):
    file_path = tmp_path / "atomic_test.json"
    atomic_write_json(file_path, {"test": "value"})
    assert file_path.exists()
    data = json.loads(file_path.read_text(encoding="utf-8"))
    assert data["test"] == "value"


def test_list_saved_runs(tmp_path, sample_envelope, sample_metrics):
    run1 = tmp_path / "run_older"
    sample_envelope.finished_at = "2026-10-07T10:00:00Z"
    save_run(run1, sample_envelope, sample_metrics, {})

    run2 = tmp_path / "run_newer"
    sample_envelope.finished_at = "2026-10-07T12:00:00Z"
    save_run(run2, sample_envelope, sample_metrics, {})

    runs = list_saved_runs(tmp_path)
    assert len(runs) == 2
    # Verify newest first
    assert runs[0].envelope.finished_at > runs[1].envelope.finished_at
