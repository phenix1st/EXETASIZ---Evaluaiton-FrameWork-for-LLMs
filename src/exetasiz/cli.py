"""Command-line interface for Exetasiz."""

from __future__ import annotations

import json
import os
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import typer
import yaml
from dotenv import load_dotenv

from exetasiz.core.dataset import DatasetLoadError, load_dataset
from exetasiz.core.models import DatasetMetadata, ResultsEnvelope
from exetasiz.core.persistence import (
    RunLoadError,
    list_saved_runs,
    load_run,
    save_run,
)
from exetasiz.evaluators.base import Evaluator
from exetasiz.evaluators.registry import get_evaluator, list_registered_evaluators
from exetasiz.metrics.metrics import (
    compare_models,
    compute_structured_metrics,
    format_comparison_table,
)
from exetasiz.providers.base import LLMProvider
from exetasiz.providers.registry import get_provider, list_registered_providers
from exetasiz.reporting.html_report import generate_html_report
from exetasiz.runner.runner import Runner
import exetasiz

load_dotenv()

app = typer.Typer(
    name="exetasiz",
    help="A framework for systematically testing and comparing LLMs on the same dataset.",
    add_completion=False,
)


@app.command("list-providers")
def list_providers_command() -> None:
    """List all registered LLM providers."""
    providers = list_registered_providers()
    typer.echo("Registered LLM Providers:")
    for p in providers:
        typer.echo(f"  * {p}")


@app.command("list-evaluators")
def list_evaluators_command() -> None:
    """List all registered evaluators."""
    evaluators = list_registered_evaluators()
    typer.echo("Registered Evaluators:")
    for e in evaluators:
        typer.echo(f"  * {e}")


@app.command("validate-dataset")
def validate_dataset_command(
    dataset_path: Path = typer.Argument(
        ...,
        help="Path to the dataset file (.json or .jsonl) to validate",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    )
) -> None:
    """Validate a dataset file for correct schema and duplicate IDs."""
    try:
        ds = load_dataset(dataset_path)
        typer.secho(f"[PASS] Dataset '{ds.name}' is valid!", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  Total test cases: {len(ds.test_cases)}")
        categories = sorted(list({tc.category for tc in ds.test_cases}))
        typer.echo(f"  Categories ({len(categories)}): {', '.join(categories)}")
    except DatasetLoadError as e:
        typer.secho(f"[FAIL] Dataset validation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("list-runs")
def list_runs_command(
    reports_dir: Path = typer.Option(
        Path("reports"),
        "--reports-dir",
        "-d",
        help="Base directory containing saved runs",
    )
) -> None:
    """List saved evaluation runs sorted newest first."""
    runs = list_saved_runs(reports_dir)
    if not runs:
        typer.echo(f"No saved runs found in '{reports_dir}'.")
        return

    import pandas as pd
    from tabulate import tabulate

    records = []
    for r in runs:
        run_folder = Path(r.run_dir).name
        dt_str = r.envelope.finished_at[:19].replace("T", " ")
        models_str = ", ".join(r.envelope.models)
        acc_pct = f"{r.metrics.overall.accuracy * 100:.1f}%"
        records.append({
            "run_directory": run_folder,
            "experiment": r.envelope.experiment_name,
            "finished_at": dt_str,
            "models": models_str,
            "accuracy": acc_pct,
        })

    df = pd.DataFrame(records)
    typer.echo(tabulate(df, headers="keys", tablefmt="simple", showindex=False))


@app.command("report")
def report_command(
    run_dir: str = typer.Argument(
        ...,
        help="Path to run directory or 'latest' for the most recent run",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination path for generated HTML report (defaults to <run_dir>/report.html)",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Open generated report in the default web browser",
    ),
    title: Optional[str] = typer.Option(
        None,
        "--title",
        "-t",
        help="Custom title for the report",
    ),
) -> None:
    """Generate a standalone HTML evaluation report from saved run results."""
    # Handle 'latest' shorthand
    if run_dir.lower() == "latest":
        saved_runs = list_saved_runs("reports")
        if not saved_runs:
            typer.secho("Error: No saved runs found in 'reports/' to select as 'latest'.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        target_dir = Path(saved_runs[0].run_dir)
    else:
        target_dir = Path(run_dir)

    try:
        loaded = load_run(target_dir)
    except RunLoadError as e:
        typer.secho(f"Error loading run: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"Unexpected error loading run '{target_dir}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        report_path = generate_html_report(loaded, output_path=output, title=title)
        typer.secho(f"[DONE] Generated HTML report at:", fg=typer.colors.GREEN, bold=True)
        typer.echo(str(report_path.resolve()))

        if open_browser:
            resolved_path = report_path.resolve()
            opened = False
            # On Windows, os.startfile launches the default registered browser reliably
            if hasattr(os, "startfile"):
                try:
                    os.startfile(str(resolved_path))
                    opened = True
                except Exception:
                    pass
            if not opened:
                try:
                    webbrowser.open(resolved_path.as_uri())
                except Exception:
                    webbrowser.open(str(resolved_path))
    except Exception as e:
        typer.secho(f"Failed to generate report: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("run")
def run_command(
    config_path: Path = typer.Argument(
        ...,
        help="Path to the experiment configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    by_category: bool = typer.Option(
        False,
        "--by-category",
        "-c",
        help="Show metrics breakdown grouped by test case category",
    ),
    report: bool = typer.Option(
        False,
        "--report",
        "-r",
        help="Automatically generate a standalone HTML report upon completion",
    ),
) -> None:
    """Run an evaluation experiment defined in a YAML configuration file."""
    started_at_dt = datetime.now(timezone.utc)
    started_at_iso = started_at_dt.isoformat()

    try:
        config_text = config_path.read_text(encoding="utf-8")
        config: Dict[str, Any] = yaml.safe_load(config_text)
    except Exception as e:
        typer.secho(f"Failed to read experiment config '{config_path}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    experiment_name = config.get("name", config_path.stem)
    dataset_rel_path = config.get("dataset")
    if not dataset_rel_path:
        typer.secho("Missing 'dataset' field in experiment config.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Resolve dataset path relative to config dir or current working dir
    candidate_path = Path(dataset_rel_path)
    if not candidate_path.is_absolute():
        if (config_path.parent / candidate_path).exists():
            dataset_file = config_path.parent / candidate_path
        else:
            dataset_file = candidate_path
    else:
        dataset_file = candidate_path

    try:
        dataset = load_dataset(dataset_file)
    except DatasetLoadError as e:
        typer.secho(f"Failed loading dataset '{dataset_file}': {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Initialize providers
    models_cfg = config.get("models", [])
    if not models_cfg:
        typer.secho("No 'models' specified in experiment config.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    providers: List[LLMProvider] = []
    for item in models_cfg:
        provider_type = item.get("provider")
        model_name = item.get("model")
        display_name = item.get("name") or model_name or provider_type
        extra_kwargs = {k: v for k, v in item.items() if k not in ("provider", "name")}
        try:
            prov = get_provider(provider_type, name=display_name, **extra_kwargs)
            providers.append(prov)
        except Exception as e:
            typer.secho(f"Failed initializing provider '{provider_type}': {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    # Initialize evaluators
    evaluators_cfg = config.get("evaluators", ["exact_match"])
    evaluators: List[Evaluator] = []
    for item in evaluators_cfg:
        if isinstance(item, str):
            eval_name = item
            eval_kwargs = {}
        elif isinstance(item, dict):
            eval_name = item.get("name") or item.get("type")
            eval_kwargs = {k: v for k, v in item.items() if k not in ("name", "type")}
        else:
            typer.secho(f"Invalid evaluator configuration: {item}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        try:
            ev = get_evaluator(eval_name, **eval_kwargs)
            evaluators.append(ev)
        except Exception as e:
            typer.secho(f"Failed initializing evaluator '{eval_name}': {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    output_dir_str = config.get("output_dir", "reports")
    base_output_dir = Path(output_dir_str)

    typer.echo(f"Starting experiment '{experiment_name}'")
    typer.echo(f"Dataset: {dataset.name} ({len(dataset.test_cases)} test cases)")
    typer.echo(f"Models: {', '.join([p.name for p in providers])}")
    typer.echo(f"Evaluators: {', '.join([e.name for e in evaluators])}")
    typer.echo("---")

    runner = Runner(
        dataset=dataset,
        providers=providers,
        evaluators=evaluators,
        experiment_name=experiment_name,
        config_snapshot=config,
    )

    total_ops = len(providers) * len(dataset.test_cases)
    with typer.progressbar(length=total_ops, label="Evaluating models") as progress:
        def on_step(current: int, total: int, prov_name: str, case_id: str) -> None:
            progress.update(1)

        run_result = runner.run(progress_callback=on_step)

    finished_at_dt = datetime.now(timezone.utc)
    finished_at_iso = finished_at_dt.isoformat()
    duration_s = max(0.001, (finished_at_dt - started_at_dt).total_seconds())

    typer.echo("\nExperiment Run Complete!\n")

    # Generate comparative metrics table
    summary_df = compare_models(run_result, by_category=by_category, test_cases=dataset.test_cases)
    table_str = format_comparison_table(summary_df)
    typer.echo(table_str)

    # Compute hierarchical StructuredMetrics
    structured_metrics = compute_structured_metrics(run_result.results, dataset.test_cases)

    # Envelope
    clean_ts = run_result.timestamp.replace(":", "-").replace(".", "-")
    run_id = f"{experiment_name}_{clean_ts}"
    run_dir = base_output_dir / run_id

    envelope = ResultsEnvelope(
        schema_version="1.0",
        run_id=run_id,
        experiment_name=experiment_name,
        started_at=started_at_iso,
        finished_at=finished_at_iso,
        duration_seconds=duration_s,
        dataset=DatasetMetadata(
            name=dataset.name,
            path=str(dataset_file),
            num_test_cases=len(dataset.test_cases),
        ),
        models=[p.name for p in providers],
        evaluators=[e.name for e in evaluators],
        exetasiz_version=getattr(exetasiz, "__version__", "0.1.0"),
        results=run_result.results,
    )

    # Save run atomically
    saved_path = save_run(
        run_dir=run_dir,
        envelope=envelope,
        metrics=structured_metrics,
        config_snapshot=config,
    )

    typer.secho(f"\n[DONE] Artifacts saved to: {saved_path}", fg=typer.colors.GREEN, bold=True)

    if report:
        loaded = load_run(saved_path)
        report_file = generate_html_report(loaded)
        typer.secho(f"[DONE] Generated HTML report at: {report_file}", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
