"""End-to-end CLI tests for reporting commands and flags."""

import json
from pathlib import Path
from typer.testing import CliRunner
import yaml

from exetasiz.cli import app

runner = CliRunner()


def _setup_mock_experiment(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps({
            "name": "cli_report_ds",
            "test_cases": [
                {"id": "c1", "category": "math", "prompt": "1+1?", "expected_answer": "2"},
                {"id": "c2", "category": "math", "prompt": "2+2?", "expected_answer": "4"},
            ],
        }),
        encoding="utf-8",
    )
    reports_dir = tmp_path / "reports"
    exp_cfg = {
        "name": "cli_report_exp",
        "dataset": str(dataset_file),
        "models": [
            {
                "provider": "mock",
                "name": "m1",
                "responses": {"1+1?": "2", "2+2?": "4"},
            }
        ],
        "evaluators": ["exact_match"],
        "output_dir": str(reports_dir),
    }
    cfg_file = tmp_path / "experiment.yaml"
    cfg_file.write_text(yaml.dump(exp_cfg), encoding="utf-8")
    return cfg_file, reports_dir


def test_cli_run_with_report_flag(tmp_path):
    cfg_file, reports_dir = _setup_mock_experiment(tmp_path)
    res = runner.invoke(app, ["run", str(cfg_file), "--report"])
    assert res.exit_code == 0
    assert "Generated HTML report at" in res.output

    saved_dirs = list(reports_dir.glob("cli_report_exp_*"))
    assert len(saved_dirs) == 1
    run_dir = saved_dirs[0]
    report_file = run_dir / "report.html"
    assert report_file.exists()
    assert report_file.stat().st_size > 0


def test_cli_report_command(tmp_path):
    cfg_file, reports_dir = _setup_mock_experiment(tmp_path)
    # Run without --report first
    runner.invoke(app, ["run", str(cfg_file)])
    saved_dirs = list(reports_dir.glob("cli_report_exp_*"))
    run_dir = saved_dirs[0]

    # Generate report via CLI command
    res = runner.invoke(app, ["report", str(run_dir)])
    assert res.exit_code == 0
    assert "Generated HTML report at:" in res.output
    assert (run_dir / "report.html").exists()


def test_cli_report_latest_and_list_runs(tmp_path):
    cfg_file, reports_dir = _setup_mock_experiment(tmp_path)
    runner.invoke(app, ["run", str(cfg_file)])

    # List runs
    list_res = runner.invoke(app, ["list-runs", "--reports-dir", str(reports_dir)])
    assert list_res.exit_code == 0
    assert "cli_report_exp" in list_res.output

    # Report latest (mocking default 'reports' by pointing or passing path)
    bad_res = runner.invoke(app, ["report", "nonexistent_dir"])
    assert bad_res.exit_code == 1
    assert "Error loading run:" in bad_res.output
