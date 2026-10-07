"""Command-line interface for Exetasiz."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import typer
import yaml
from dotenv import load_dotenv

from exetasiz.core.dataset import DatasetLoadError, load_dataset
from exetasiz.evaluators.base import Evaluator
from exetasiz.evaluators.registry import get_evaluator, list_registered_evaluators
from exetasiz.metrics.metrics import compare_models, format_comparison_table
from exetasiz.providers.base import LLMProvider
from exetasiz.providers.registry import get_provider, list_registered_providers
from exetasiz.runner.runner import Runner

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
) -> None:
    """Run an evaluation experiment defined in a YAML configuration file."""
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

    typer.echo("\nExperiment Run Complete!\n")

    # Generate metrics
    summary_df = compare_models(run_result, by_category=by_category, test_cases=dataset.test_cases)
    table_str = format_comparison_table(summary_df)
    typer.echo(table_str)

    # Save reports to reports/<experiment>_<timestamp_sanitized>/
    clean_ts = run_result.timestamp.replace(":", "-").replace(".", "-")
    run_dir = base_output_dir / f"{experiment_name}_{clean_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.yaml
    (run_dir / "config.yaml").write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")

    # 2. results.json
    (run_dir / "results.json").write_text(
        run_result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # 3. metrics.json
    metrics_records = summary_df.to_dict(orient="records")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_records, indent=2),
        encoding="utf-8",
    )

    typer.secho(f"\n[DONE] Artifacts saved to: {run_dir}", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
