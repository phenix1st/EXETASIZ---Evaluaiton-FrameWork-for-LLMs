"""End-to-end CLI tests using Typer CliRunner."""

import json
from pathlib import Path
from typer.testing import CliRunner
import yaml

from exetasiz.cli import app

runner = CliRunner()


def test_cli_list_providers():
    result = runner.invoke(app, ["list-providers"])
    assert result.exit_code == 0
    assert "mock" in result.output
    assert "openai" in result.output
    assert "gemini" in result.output
    assert "local" in result.output


def test_cli_list_evaluators():
    result = runner.invoke(app, ["list-evaluators"])
    assert result.exit_code == 0
    assert "exact_match" in result.output
    assert "semantic_similarity" in result.output


def test_cli_validate_dataset_success(tmp_path):
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(
        json.dumps({
            "name": "test_ds",
            "test_cases": [
                {"id": "1", "category": "general", "prompt": "hi", "expected_answer": "hello"}
            ],
        }),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate-dataset", str(dataset_file)])
    assert result.exit_code == 0
    assert "Dataset 'test_ds' is valid!" in result.output


def test_cli_validate_dataset_failure(tmp_path):
    dataset_file = tmp_path / "dup.json"
    dataset_file.write_text(
        json.dumps([
            {"id": "dup", "category": "general", "prompt": "hi", "expected_answer": "hello"},
            {"id": "dup", "category": "general", "prompt": "hi2", "expected_answer": "hello2"},
        ]),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate-dataset", str(dataset_file)])
    assert result.exit_code == 1
    assert "Duplicate test case IDs" in result.output


def test_cli_run_e2e(tmp_path):
    # Setup temporary dataset
    dataset_file = tmp_path / "qa.json"
    dataset_file.write_text(
        json.dumps({
            "name": "qa_e2e",
            "test_cases": [
                {"id": "1", "category": "reasoning", "prompt": "15% of 240?", "expected_answer": "36"},
                {"id": "2", "category": "factual", "prompt": "Capital of France?", "expected_answer": "Paris"},
            ],
        }),
        encoding="utf-8",
    )

    reports_dir = tmp_path / "reports"

    # Setup experiment YAML
    exp_config = {
        "name": "cli_e2e_experiment",
        "dataset": str(dataset_file),
        "models": [
            {
                "provider": "mock",
                "name": "mock-perfect",
                "responses": {
                    "15% of 240?": "36",
                    "Capital of France?": "Paris",
                },
            },
            {
                "provider": "mock",
                "name": "mock-flawed",
                "responses": {
                    "15% of 240?": "42",
                    "Capital of France?": "Paris",
                },
            },
        ],
        "evaluators": ["exact_match"],
        "output_dir": str(reports_dir),
    }

    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(yaml.dump(exp_config), encoding="utf-8")

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "mock-perfect" in result.output
    assert "mock-flawed" in result.output

    # Check generated output files
    saved_dirs = list(reports_dir.glob("cli_e2e_experiment_*"))
    assert len(saved_dirs) == 1
    run_dir = saved_dirs[0]

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "metrics.json").exists()

    # Validate metrics content
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics) == 2
    perfect_metric = next(m for m in metrics if m["model"] == "mock-perfect")
    flawed_metric = next(m for m in metrics if m["model"] == "mock-flawed")

    assert perfect_metric["accuracy"] == 1.0
    assert flawed_metric["accuracy"] == 0.5
